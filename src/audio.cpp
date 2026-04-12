#include "audio.h"

#include <aubio/aubio.h>
#include <sndfile.h>
#include <samplerate.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Snap a detected frequency to the nearest octave multiple of target_freq.
// aubio yinfft frequently returns the sub-octave (f/2) or super-octave (f*2)
// instead of the true fundamental.  Because the caller always knows the
// expected pitch (the MIDI note number), we can use it to disambiguate.
// After snapping, a 200-cent gate is still applied to reject genuinely bad
// detections (noise, unvoiced frames, etc.).
static double snap_to_octave(double f, double target_freq)
{
    if (f <= 0.0 || target_freq <= 0.0) return f;
    // Number of octaves to shift: round log2(f / target) to the nearest integer
    double octaves = std::round(std::log2(f / target_freq));
    return f / std::pow(2.0, octaves);
}

// Analysis window size that allows detection down to ~20 Hz.
// Minimum detectable frequency = sample_rate / (window_size / 2)
// We round up to the nearest multiple of 512 for alignment.
static int analysis_window_for_rate(int sample_rate)
{
    // We need: window_size / 2 >= sample_rate / 20
    int min_half = static_cast<int>(std::ceil(sample_rate / 20.0));
    int half = ((min_half + 511) / 512) * 512; // round up to multiple of 512
    return half * 2;
}

// Extract first channel from interleaved buffer.
static std::vector<float> to_mono(const std::vector<float>& interleaved,
                                   int channels, long long frames)
{
    std::vector<float> mono(static_cast<size_t>(frames));
    for (long long i = 0; i < frames; ++i)
        mono[static_cast<size_t>(i)] = interleaved[static_cast<size_t>(i * channels)];
    return mono;
}

// Resample a segment of interleaved audio.
// src_ratio = output_frames / input_frames (libsamplerate convention).
static std::vector<float> resample_segment(const float* input,
                                           long long input_frames,
                                           int channels,
                                           double src_ratio)
{
    if (input_frames <= 0) return {};

    long long output_frames = static_cast<long long>(
        std::llround(static_cast<double>(input_frames) * src_ratio));
    if (output_frames < 1) output_frames = 1;

    std::vector<float> output(static_cast<size_t>(output_frames * channels), 0.0f);

    SRC_DATA sd = {};
    sd.data_in       = input;
    sd.input_frames  = static_cast<long>(input_frames);
    sd.data_out      = output.data();
    sd.output_frames = static_cast<long>(output_frames);
    sd.src_ratio     = src_ratio;
    sd.end_of_input  = 1;

    int err = src_simple(&sd, SRC_SINC_BEST_QUALITY, channels);
    if (err != 0)
        std::cerr << "Warning: resampling error: " << src_strerror(err) << "\n";

    // Ensure vector is exactly the expected size
    output.resize(static_cast<size_t>(output_frames * channels), 0.0f);
    return output;
}

// ---------------------------------------------------------------------------
// Audio I/O
// ---------------------------------------------------------------------------

bool load_audio(const std::string& path,
                std::vector<float>& samples,
                int&       sample_rate,
                int&       channels,
                long long& total_frames,
                int&       format)
{
    SF_INFO info = {};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf) {
        std::cerr << "Error opening '" << path << "': " << sf_strerror(nullptr) << "\n";
        return false;
    }

    sample_rate  = info.samplerate;
    channels     = info.channels;
    total_frames = info.frames;
    format       = info.format;

    samples.resize(static_cast<size_t>(total_frames * channels));
    sf_count_t got = sf_readf_float(sf, samples.data(), total_frames);
    sf_close(sf);

    if (got != total_frames) {
        std::cerr << "Warning: read " << got << " of " << total_frames
                  << " frames from '" << path << "'\n";
        total_frames = got;
        samples.resize(static_cast<size_t>(total_frames * channels));
    }
    return true;
}

bool save_audio(const std::string& path,
                const std::vector<float>& samples,
                int sample_rate,
                int channels,
                int format)
{
    SF_INFO info = {};
    info.samplerate = sample_rate;
    info.channels   = channels;
    info.format     = format;

    if (!sf_format_check(&info)) {
        // Fall back to 24-bit WAV if the original format can't be written
        info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_24;
        std::cerr << "Warning: original format not writable, falling back to 24-bit WAV\n";
    }

    SNDFILE* sf = sf_open(path.c_str(), SFM_WRITE, &info);
    if (!sf) {
        std::cerr << "Error creating '" << path << "': " << sf_strerror(nullptr) << "\n";
        return false;
    }

    long long total_frames = static_cast<long long>(samples.size()) / channels;
    sf_count_t written = sf_writef_float(sf, samples.data(), total_frames);
    sf_close(sf);

    if (written != total_frames) {
        std::cerr << "Warning: wrote " << written << " of " << total_frames
                  << " frames to '" << path << "'\n";
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pitch analysis
// ---------------------------------------------------------------------------

double detect_overall_pitch(const std::vector<float>& samples,
                            int       channels,
                            int       sample_rate,
                            double    target_freq,
                            float     yin_threshold,
                            bool      warn_bimodal,
                            double*   secondary_out)
{
    long long total_frames = static_cast<long long>(samples.size()) / channels;
    std::vector<float> mono = to_mono(samples, channels, total_frames);

    uint_t win = static_cast<uint_t>(analysis_window_for_rate(sample_rate));

    // For high-pitched targets the default window (up to 5120 frames at 48 kHz)
    // can be larger than the entire note body of a short staccatissimo sample,
    // leaving too few hops in the counted region.  Scale the window down so it
    // spans at most ~15 complete periods of the target pitch.  Round to the
    // nearest 256-frame boundary and never go below 512.  Low-pitched targets
    // are unaffected.
    if (target_freq > 20.0) {
        const int kMinPeriods = 15;
        auto target_half = static_cast<uint_t>(
            std::ceil(kMinPeriods * static_cast<double>(sample_rate) / target_freq));
        target_half = std::max(static_cast<uint_t>(256),
                               ((target_half + 255) / 256) * 256);
        win = std::min(win, target_half * 2);
    }

    uint_t hop = win / 2;

    // Skip the first 5% of frames to avoid attack transients, but run the
    // detector from pos=0 so its internal autocorrelation buffer is fully
    // populated before we start counting results.  The end-of-file tail skip
    // is intentionally omitted: the median is robust to a handful of release
    // detections, and for very short samples (staccatissimo < 0.5 s) the end
    // skip would discard hops we can't afford to lose.
    long long skip           = total_frames / 20;
    long long analysis_start = skip;

    // aubio_pitch_get_confidence() returns 0.0 for "yinfft" in many aubio builds,
    // so we use a cents gate (after octave snapping) as the quality filter.
    // The tolerance controls the YIN aperiodicity threshold — lower is stricter
    // and will return 0 Hz for breathy/noisy frames.  We try progressively more
    // permissive settings so that vocal whistles, breathy flutes, etc. still work.
    // gate_cents: radius around target (post-octave-snap); 200 cents = 2 semitones.
    //             Octave errors (1200 cents) are rejected even before the gate by
    //             snap_to_octave(), so 200 cents is enough for normally-tuned samples
    //             while still rejecting spurious off-frequency detections.
    auto collect = [&](float tol, double gate_cents) {
        aubio_pitch_t* pd = new_aubio_pitch("yinfft", win, hop,
                                             static_cast<uint_t>(sample_rate));
        aubio_pitch_set_unit(pd, "Hz");
        aubio_pitch_set_tolerance(pd, tol);
        aubio_pitch_set_silence(pd, -80.f);

        fvec_t* ibuf = new_fvec(hop);
        fvec_t* obuf = new_fvec(1);

        std::vector<float> freqs;
        for (long long pos = 0;
             pos + static_cast<long long>(hop) <= total_frames;
             pos += static_cast<long long>(hop))
        {
            for (uint_t i = 0; i < hop; ++i)
                ibuf->data[i] = mono[static_cast<size_t>(pos + i)];

            aubio_pitch_do(pd, ibuf, obuf);

            if (pos < analysis_start) continue; // attack warmup, discard

            float f = obuf->data[0];

            if (f > 15.f && f < 22000.f) {
                double snapped = snap_to_octave(static_cast<double>(f), target_freq);
                double cents   = std::abs(1200.0 * std::log2(snapped / target_freq));
                if (cents <= gate_cents)
                    freqs.push_back(static_cast<float>(snapped));
            }
        }

        del_fvec(ibuf);
        del_fvec(obuf);
        del_aubio_pitch(pd);
        return freqs;
    };

    // Pass 1: user-supplied tolerance, ±400-cent gate (4 semitones).
    std::vector<float> freqs = collect(yin_threshold, 200.0);
    // Pass 2: relax tolerance for breathy/complex timbres (e.g. vocal whistles).
    if (freqs.empty()) freqs = collect(std::min(yin_threshold * 2.0f, 0.5f), 200.0);

    if (freqs.empty()) return 0.0;

    std::sort(freqs.begin(), freqs.end());

    // --- Bimodality check ---
    // If the sorted pitch estimates have a clear gap (> 15 cents) with enough
    // samples on both sides, the recording likely contains two simultaneous pitch
    // components (e.g. a vocal whistle where the voice and the whistle both register).
    // Detection always runs; printing is conditional on warn_bimodal.
    if (freqs.size() >= 6) {
        const double kBimodalGapCents = 15.0;
        const size_t kMinClusterSize  = 3;

        double max_gap = 0.0;
        size_t gap_pos = 0;
        for (size_t i = 1; i < freqs.size(); ++i) {
            double gap = 1200.0 * std::log2(static_cast<double>(freqs[i]) /
                                             static_cast<double>(freqs[i-1]));
            if (gap > max_gap) { max_gap = gap; gap_pos = i; }
        }

        size_t n_low  = gap_pos;
        size_t n_high = freqs.size() - gap_pos;

        if (max_gap >= kBimodalGapCents &&
            n_low  >= kMinClusterSize   &&
            n_high >= kMinClusterSize)
        {
            double low_hz  = static_cast<double>(freqs[(n_low  - 1) / 2]);
            double high_hz = static_cast<double>(freqs[gap_pos + (n_high - 1) / 2]);
            double low_c   = 1200.0 * std::log2(low_hz  / target_freq);
            double high_c  = 1200.0 * std::log2(high_hz / target_freq);

            // Overall median determines the dominant cluster
            bool   dom_is_low = (freqs.size() / 2 < gap_pos);
            double dom_hz  = dom_is_low ? low_hz  : high_hz;
            double alt_hz  = dom_is_low ? high_hz : low_hz;
            double dom_c   = dom_is_low ? low_c   : high_c;
            double alt_c   = dom_is_low ? high_c  : low_c;
            size_t dom_cnt = dom_is_low ? n_low   : n_high;
            size_t alt_cnt = dom_is_low ? n_high  : n_low;

            if (secondary_out) *secondary_out = alt_hz;

            if (warn_bimodal) {
                std::cerr << std::fixed << std::setprecision(1)
                          << "Warning: two pitch components detected.\n"
                          << "  Dominant  : " << dom_hz << " Hz  ("
                          << (dom_c >= 0 ? "+" : "") << dom_c << "c)  ["
                          << dom_cnt << " frames]\n"
                          << "  Secondary : " << alt_hz << " Hz  ("
                          << (alt_c >= 0 ? "+" : "") << alt_c << "c)  ["
                          << alt_cnt << " frames]\n";
                if (secondary_out)
                    std::cerr << "  Auto-selecting secondary (-s flag).\n";
                else
                    std::cerr << "  Using dominant. If secondary is the target "
                                 "component, re-run with: -f " << alt_hz << "\n"
                              << "  Or use -s to auto-select the secondary "
                                 "whenever two components are detected.\n";
            }
        }
    }

    return static_cast<double>(freqs[freqs.size() / 2]); // median
}

// ---------------------------------------------------------------------------
// Drift correction
// ---------------------------------------------------------------------------

std::vector<SegmentCorrection> compute_drift_corrections(
    const std::vector<float>& samples,
    int    channels,
    int    sample_rate,
    double target_freq,
    float  yin_threshold,
    double* out_global_freq,
    bool    warn_bimodal)
{
    long long total_frames = static_cast<long long>(samples.size()) / channels;
    std::vector<float> mono = to_mono(samples, channels, total_frames);

    uint_t win = static_cast<uint_t>(analysis_window_for_rate(sample_rate));
    uint_t hop = win / 2;
    int    seg = static_cast<int>(win); // one correction segment per analysis window

    long long num_segs = (total_frames + seg - 1) / seg;

    // --- Pass 1: robust global pitch estimate ---
    // detect_overall_pitch() skips the first/last 5% of the file and returns
    // the median of all accepted detections.  This is resistant to attack /
    // release transients, which are the main source of false readings.
    // We use this as the centre of the gate in Pass 2 instead of target_freq.
    // If global detection fails we fall back to target_freq.
    double global_freq = detect_overall_pitch(samples, channels, sample_rate,
                                               target_freq, yin_threshold,
                                               warn_bimodal);
    if (global_freq <= 0.0) global_freq = target_freq;
    if (out_global_freq) *out_global_freq = global_freq;

    // --- Pass 2: per-segment detection, gate centred on global_freq ---
    // Using global_freq as the gate centre prevents attack/release artefacts
    // from being accepted: they are typically farther from the stable sustained
    // pitch than from the target note.
    // 150 cents ≈ 1.5 semitones — enough for real intonation drift, but tight
    // enough to reject transient detections that would cause a pitch ramp.
    const double kGateCents = 150.0;

    auto run_pass = [&](float tol, std::vector<std::vector<double>>& out)
    {
        aubio_pitch_t* pd = new_aubio_pitch("yinfft", win, hop,
                                             static_cast<uint_t>(sample_rate));
        aubio_pitch_set_unit(pd, "Hz");
        aubio_pitch_set_tolerance(pd, tol);
        aubio_pitch_set_silence(pd, -80.f);

        fvec_t* ibuf = new_fvec(hop);
        fvec_t* obuf = new_fvec(1);

        for (long long pos = 0;
             pos + static_cast<long long>(hop) <= total_frames;
             pos += static_cast<long long>(hop))
        {
            for (uint_t i = 0; i < hop; ++i)
                ibuf->data[i] = mono[static_cast<size_t>(pos + i)];

            aubio_pitch_do(pd, ibuf, obuf);
            float f = obuf->data[0];

            if (f > 15.f && f < 22000.f) {
                double snapped = snap_to_octave(static_cast<double>(f), target_freq);
                double cents   = std::abs(1200.0 * std::log2(snapped / global_freq));
                if (cents <= kGateCents) {
                    long long seg_idx = pos / seg;
                    if (seg_idx < num_segs)
                        out[static_cast<size_t>(seg_idx)].push_back(snapped);
                }
            }
        }

        del_fvec(ibuf);
        del_fvec(obuf);
        del_aubio_pitch(pd);
    };

    std::vector<std::vector<double>> seg_freqs(static_cast<size_t>(num_segs));
    run_pass(yin_threshold, seg_freqs);

    // If fewer than half the segments have data, retry with relaxed tolerance.
    {
        long long covered = 0;
        for (const auto& v : seg_freqs) if (!v.empty()) ++covered;
        if (covered < num_segs / 2) {
            seg_freqs.assign(static_cast<size_t>(num_segs), {});
            run_pass(std::min(yin_threshold * 2.0f, 0.5f), seg_freqs);
        }
    }

    // Reduce each segment to a single median value.
    std::vector<double> raw(static_cast<size_t>(num_segs), 0.0);
    for (long long s = 0; s < num_segs; ++s) {
        auto& v = seg_freqs[static_cast<size_t>(s)];
        if (!v.empty()) {
            std::sort(v.begin(), v.end());
            raw[static_cast<size_t>(s)] = v[v.size() / 2];
        }
    }

    // --- Fill missing segments with global_freq ---
    // Segments with no valid local estimate (attack, release, heavily breathy
    // sections) receive the global detected pitch.  This gives those segments a
    // neutral correction equal to the overall shift, rather than a spurious
    // per-segment drift based on interpolated neighbours.
    bool any_valid = false;
    for (long long s = 0; s < num_segs; ++s) {
        if (raw[static_cast<size_t>(s)] > 0.0) { any_valid = true; break; }
    }
    if (!any_valid) {
        std::cerr << "Error: no valid pitch detected for drift correction\n";
        return {};
    }
    for (long long s = 0; s < num_segs; ++s)
        if (raw[static_cast<size_t>(s)] <= 0.0)
            raw[static_cast<size_t>(s)] = global_freq;

    // --- Smooth with a 7-point moving average ---
    // At 48 kHz the window spans ~742 ms, covering ~4 vibrato cycles at 6 Hz,
    // which averages out vibrato while still tracking slow intonation drift.
    const int R = 3; // radius
    std::vector<double> smooth(static_cast<size_t>(num_segs));
    for (long long s = 0; s < num_segs; ++s) {
        long long lo = std::max(0LL, s - R);
        long long hi = std::min(num_segs - 1, s + R);
        double sum = 0.0;
        int cnt = 0;
        for (long long k = lo; k <= hi; ++k) {
            double v = raw[static_cast<size_t>(k)];
            if (v > 0.0) { sum += v; ++cnt; }
        }
        smooth[static_cast<size_t>(s)] = (cnt > 0) ? (sum / cnt) : global_freq;
    }

    // --- Build SegmentCorrection list ---
    // Per-segment corrections are only applied when the segment's smoothed pitch
    // deviates from global_freq by more than kMinDriftCents.  Below that threshold
    // the deviation is indistinguishable from detection noise (~8-10 cents RMS
    // after the 7-point smoother), and applying it would make the output less
    // accurate than the input.  Those segments instead receive only the global
    // correction (global_freq → target_freq), which is computed from the robust
    // median and is always safe to apply.
    const double kMinDriftCents = 20.0;

    std::vector<SegmentCorrection> corrections;
    corrections.reserve(static_cast<size_t>(num_segs));

    for (long long s = 0; s < num_segs; ++s) {
        long long start = s * seg;
        long long len   = std::min(static_cast<long long>(seg), total_frames - start);

        double detected = smooth[static_cast<size_t>(s)];
        if (detected < 1.0) detected = global_freq; // safety

        // If the per-segment deviation from the global pitch is below the noise
        // floor, use global_freq so only the overall pitch shift is applied.
        double dev_from_global = std::abs(1200.0 * std::log2(detected / global_freq));
        if (dev_from_global < kMinDriftCents)
            detected = global_freq;

        double ratio = detected / target_freq;
        // Clamp to ±6 semitones (factor ~0.707–1.414) to avoid extreme artefacts
        ratio = std::max(0.707, std::min(1.414, ratio));

        long long out_len = std::llround(static_cast<double>(len) * ratio);
        if (out_len < 1) out_len = 1;

        corrections.push_back({ start, len, out_len, ratio });
    }

    return corrections;
}

// ---------------------------------------------------------------------------
// Diagnostic analysis
// ---------------------------------------------------------------------------

void analyze_audio(const std::vector<float>& samples,
                   int    channels,
                   int    sample_rate,
                   double target_freq,
                   float  yin_threshold)
{
    long long total_frames = static_cast<long long>(samples.size()) / channels;
    std::vector<float> mono = to_mono(samples, channels, total_frames);

    uint_t win = static_cast<uint_t>(analysis_window_for_rate(sample_rate));
    uint_t hop = win / 2;
    int    seg = static_cast<int>(win);
    long long num_segs = (total_frames + seg - 1) / seg;

    double seg_secs = static_cast<double>(seg) / sample_rate;

    // --- Pass 1: global pitch ---
    double global_freq = detect_overall_pitch(samples, channels, sample_rate,
                                               target_freq, yin_threshold);
    double global_cents = (global_freq > 0.0)
        ? 1200.0 * std::log2(global_freq / target_freq)
        : 0.0;

    std::cout << "\n--- Pass 1: global pitch ---\n";
    if (global_freq <= 0.0) {
        std::cout << "  FAILED – no valid pitch detected in stable region\n";
        global_freq = target_freq;
    } else {
        std::cout << "  Detected  : " << global_freq << " Hz\n";
        std::cout << "  vs target : " << target_freq << " Hz  ("
                  << (global_cents >= 0 ? "+" : "") << global_cents << " cents)\n";
    }

    // --- Pass 2: per-segment detection ---
    const double kGateCents   = 150.0;
    const double kMinDriftCents = 20.0;

    auto run_pass = [&](float tol, std::vector<std::vector<double>>& out) {
        aubio_pitch_t* pd = new_aubio_pitch("yinfft", win, hop,
                                             static_cast<uint_t>(sample_rate));
        aubio_pitch_set_unit(pd, "Hz");
        aubio_pitch_set_tolerance(pd, tol);
        aubio_pitch_set_silence(pd, -80.f);
        fvec_t* ibuf = new_fvec(hop);
        fvec_t* obuf = new_fvec(1);

        for (long long pos = 0;
             pos + static_cast<long long>(hop) <= total_frames;
             pos += static_cast<long long>(hop))
        {
            for (uint_t i = 0; i < hop; ++i)
                ibuf->data[i] = mono[static_cast<size_t>(pos + i)];
            aubio_pitch_do(pd, ibuf, obuf);
            float f = obuf->data[0];
            if (f > 15.f && f < 22000.f) {
                double snapped = snap_to_octave(static_cast<double>(f), target_freq);
                if (std::abs(1200.0 * std::log2(snapped / global_freq)) <= kGateCents) {
                    long long idx = pos / seg;
                    if (idx < num_segs)
                        out[static_cast<size_t>(idx)].push_back(snapped);
                }
            }
        }
        del_fvec(ibuf); del_fvec(obuf); del_aubio_pitch(pd);
    };

    std::vector<std::vector<double>> seg_freqs(static_cast<size_t>(num_segs));
    run_pass(yin_threshold, seg_freqs);

    long long covered = 0;
    for (const auto& v : seg_freqs) if (!v.empty()) ++covered;
    bool used_fallback = false;
    if (covered < num_segs / 2) {
        used_fallback = true;
        seg_freqs.assign(static_cast<size_t>(num_segs), {});
        run_pass(std::min(yin_threshold * 2.0f, 0.5f), seg_freqs);
        covered = 0;
        for (const auto& v : seg_freqs) if (!v.empty()) ++covered;
    }

    // Reduce to medians
    std::vector<double> raw(static_cast<size_t>(num_segs), 0.0);
    for (long long s = 0; s < num_segs; ++s) {
        auto& v = seg_freqs[static_cast<size_t>(s)];
        if (!v.empty()) {
            std::sort(v.begin(), v.end());
            raw[static_cast<size_t>(s)] = v[v.size() / 2];
        }
    }

    // Fill missing with global
    for (long long s = 0; s < num_segs; ++s)
        if (raw[static_cast<size_t>(s)] <= 0.0)
            raw[static_cast<size_t>(s)] = global_freq;

    // Smooth
    const int R = 3;
    std::vector<double> smooth(static_cast<size_t>(num_segs));
    for (long long s = 0; s < num_segs; ++s) {
        long long lo = std::max(0LL, s - R), hi = std::min(num_segs - 1, s + R);
        double sum = 0.0; int cnt = 0;
        for (long long k = lo; k <= hi; ++k) {
            double v = raw[static_cast<size_t>(k)];
            if (v > 0.0) { sum += v; ++cnt; }
        }
        smooth[static_cast<size_t>(s)] = (cnt > 0) ? sum / cnt : global_freq;
    }

    std::cout << "\n--- Pass 2: per-segment detection ---\n";
    std::cout << "  Window    : " << win << " samples = " << seg_secs << " s\n";
    std::cout << "  Segments  : " << num_segs << " total, " << covered
              << " valid detections\n";
    std::cout << "  Tolerance : " << (used_fallback ? "relaxed (fallback)" : "normal")
              << "  (yin_threshold="
              << (used_fallback ? std::min(yin_threshold * 2.0f, 0.5f) : yin_threshold)
              << ")\n\n";

    // "shift"      – no measurable drift; global correction applied uniformly
    // "shift+drift" – segment deviates enough from global to warrant extra fine-tuning
    std::cout << "  Seg  Time      Raw Hz    Smooth Hz  ΔGlobal    ΔTarget    Action\n";
    std::cout << "  ---  --------  --------  ---------  ---------  ---------  ------\n";

    int n_corrected = 0;
    double min_smooth = 1e9, max_smooth = 0.0;
    for (long long s = 0; s < num_segs; ++s) {
        double time_s     = static_cast<double>(s) * seg_secs;
        double raw_val    = raw[static_cast<size_t>(s)];
        double smooth_val = smooth[static_cast<size_t>(s)];
        bool   was_filled = seg_freqs[static_cast<size_t>(s)].empty();
        double dev_global = 1200.0 * std::log2(smooth_val / global_freq);
        double dev_target = 1200.0 * std::log2(smooth_val / target_freq);

        const char* action;
        if (std::abs(dev_global) >= kMinDriftCents) {
            action = "shift+drift";
            ++n_corrected;
            min_smooth = std::min(min_smooth, smooth_val);
            max_smooth = std::max(max_smooth, smooth_val);
        } else {
            action = "shift";
        }

        // Only print every segment (they may be many; truncate for very long files)
        if (num_segs <= 120 || s < 5 || s >= num_segs - 5 ||
            std::abs(dev_global) >= kMinDriftCents)
        {
            std::cout << "  " << std::setw(3) << s
                      << "  " << std::fixed << std::setprecision(3)
                      << std::setw(7) << time_s << "s"
                      << "  " << std::setw(8) << std::setprecision(2)
                      << (was_filled ? 0.0 : raw_val)
                      << "  " << std::setw(9) << smooth_val
                      << "  " << std::showpos << std::setw(8) << std::setprecision(1)
                      << dev_global << "c"
                      << "  " << std::setw(8) << dev_target << "c"
                      << std::noshowpos
                      << "  " << action
                      << (was_filled ? "  [no detection]" : "")
                      << "\n";
        } else if (s == 5 && num_segs > 120) {
            std::cout << "  ... (" << num_segs - 10 << " segments omitted)\n";
        }
    }

    double applied_drift = (n_corrected > 1)
        ? 1200.0 * std::log2(max_smooth / min_smooth) : 0.0;

    double global_ratio = (global_freq > 0.0) ? global_freq / target_freq : 1.0;

    std::cout << "\n--- Summary ---\n";
    std::cout << "  Global correction   : "
              << (global_cents >= 0 ? "+" : "") << global_cents
              << " cents  (ratio " << std::fixed << std::setprecision(5) << global_ratio << ")"
              << "  →  will be applied to ALL " << num_segs << " segments\n";
    std::cout << "  Per-segment drift   : " << std::setprecision(1)
              << applied_drift << " cents peak-to-peak"
              << "  (" << n_corrected << " / " << num_segs
              << " segments have additional drift correction)\n";
    if (n_corrected == 0 && std::abs(global_cents) > 0.01) {
        std::cout << "  Note: no pitch drift detected — the "
                  << (global_cents >= 0 ? "+" : "") << global_cents
                  << "-cent global correction WILL be applied uniformly to all segments.\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Correction application
// ---------------------------------------------------------------------------

std::vector<float> apply_corrections(const std::vector<float>& samples,
                                     int channels,
                                     const std::vector<SegmentCorrection>& corrections)
{
    long long total_input = static_cast<long long>(samples.size()) / channels;

    // Pre-estimate output size
    long long est = 0;
    for (const auto& c : corrections) est += c.output_frames;
    std::vector<float> output;
    output.reserve(static_cast<size_t>(est * channels));

    for (const auto& c : corrections) {
        if (c.input_offset >= total_input) break;

        long long in_len = c.input_frames;
        if (c.input_offset + in_len > total_input)
            in_len = total_input - c.input_offset;
        if (in_len <= 0) continue;

        const float* ptr = samples.data() +
                           static_cast<size_t>(c.input_offset * channels);

        std::vector<float> seg = resample_segment(ptr, in_len, channels, c.src_ratio);
        output.insert(output.end(), seg.begin(), seg.end());
    }

    // If the input file is longer than what the corrections cover, append the
    // remainder unchanged (shouldn't happen in normal usage but safe to handle).
    long long covered = 0;
    for (const auto& c : corrections)
        covered = std::max(covered, c.input_offset + c.input_frames);
    if (covered < total_input) {
        const float* tail = samples.data() + static_cast<size_t>(covered * channels);
        long long tail_len = (total_input - covered) * channels;
        output.insert(output.end(), tail, tail + tail_len);
    }

    return output;
}
