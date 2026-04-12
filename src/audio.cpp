#include "audio.h"
#include "yin.h"

#include <sndfile.h>
#include <samplerate.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Choose a YIN window size that allows detection down to ~20 Hz.
// Minimum detectable frequency = sample_rate / (window_size / 2)
// We round up to the nearest multiple of 512 for alignment.
static int yin_window_for_rate(int sample_rate)
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
                            float     yin_threshold)
{
    long long total_frames = static_cast<long long>(samples.size()) / channels;
    std::vector<float> mono = to_mono(samples, channels, total_frames);

    int win = yin_window_for_rate(sample_rate);
    YIN yin(sample_rate, win, yin_threshold);
    int hop = win / 2;

    // Skip the first and last 5% to avoid transient and release artefacts
    long long skip = total_frames / 20;
    long long analysis_start = skip;
    long long analysis_end   = total_frames - skip;

    auto collect = [&](float conf_threshold) {
        std::vector<float> freqs;
        for (long long pos = analysis_start;
             pos + win <= analysis_end;
             pos += hop)
        {
            float conf = 0.0f;
            float f = yin.detect(mono.data() + pos, win, &conf);
            if (f > 15.0f && f < 22000.0f && conf >= conf_threshold) {
                // Accept only detections within 2 semitones of target.
                // The target is known, so anything further away is a bad detection.
                double cents = std::abs(1200.0 * std::log2(
                    static_cast<double>(f) / target_freq));
                if (cents <= 200.0)
                    freqs.push_back(f);
            }
        }
        return freqs;
    };

    std::vector<float> freqs = collect(0.4f);
    if (freqs.empty()) freqs = collect(0.2f); // relax if nothing found (e.g. breathy timbres)

    if (freqs.empty()) return 0.0;

    std::sort(freqs.begin(), freqs.end());
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
    float  yin_threshold)
{
    long long total_frames = static_cast<long long>(samples.size()) / channels;
    std::vector<float> mono = to_mono(samples, channels, total_frames);

    int win = yin_window_for_rate(sample_rate);
    int seg = win; // one correction segment per YIN window (~106 ms at 48 kHz)
    YIN yin(sample_rate, win, yin_threshold);

    long long num_segs = (total_frames + seg - 1) / seg;

    // --- Collect raw frequency estimates per segment ---
    std::vector<double> raw(static_cast<size_t>(num_segs), 0.0);
    for (long long s = 0; s < num_segs; ++s) {
        long long start = s * seg;
        long long len   = std::min(static_cast<long long>(seg), total_frames - start);
        if (len < win) continue;

        float conf = 0.0f;
        float f = yin.detect(mono.data() + start, win, &conf);
        if (f > 15.0f && f < 22000.0f && conf > 0.2f) {
            double freq  = static_cast<double>(f);
            double cents = std::abs(1200.0 * std::log2(freq / target_freq));
            if (cents <= 200.0)
                raw[static_cast<size_t>(s)] = freq;
        }
    }

    // --- Interpolate missing estimates ---
    {
        int first_valid = -1, last_valid = -1;
        for (int s = 0; s < static_cast<int>(num_segs); ++s) {
            if (raw[static_cast<size_t>(s)] > 0.0) {
                if (first_valid < 0) first_valid = s;
                last_valid = s;
            }
        }
        if (first_valid < 0) {
            std::cerr << "Error: no valid pitch detected for drift correction\n";
            return {};
        }

        // Extend edges
        for (int s = 0; s < first_valid; ++s)
            raw[static_cast<size_t>(s)] = raw[static_cast<size_t>(first_valid)];
        for (int s = last_valid + 1; s < static_cast<int>(num_segs); ++s)
            raw[static_cast<size_t>(s)] = raw[static_cast<size_t>(last_valid)];

        // Linear interpolation between valid points
        int prev = first_valid;
        for (int s = first_valid + 1; s <= last_valid; ++s) {
            if (raw[static_cast<size_t>(s)] > 0.0) {
                if (s - prev > 1) {
                    for (int k = prev + 1; k < s; ++k) {
                        double t = static_cast<double>(k - prev) / (s - prev);
                        raw[static_cast<size_t>(k)] =
                            raw[static_cast<size_t>(prev)] * (1.0 - t) +
                            raw[static_cast<size_t>(s)]    * t;
                    }
                }
                prev = s;
            }
        }
    }

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
        smooth[static_cast<size_t>(s)] = (cnt > 0) ? (sum / cnt) : target_freq;
    }

    // --- Build SegmentCorrection list ---
    std::vector<SegmentCorrection> corrections;
    corrections.reserve(static_cast<size_t>(num_segs));

    for (long long s = 0; s < num_segs; ++s) {
        long long start = s * seg;
        long long len   = std::min(static_cast<long long>(seg), total_frames - start);

        double detected = smooth[static_cast<size_t>(s)];
        if (detected < 1.0) detected = target_freq; // safety

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
