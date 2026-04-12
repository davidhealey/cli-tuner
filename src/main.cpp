#include "audio.h"

#include <cmath>
#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog
        << " -n <midi_note> -o <output_dir> [options] file1 [file2 ...]\n\n"
        << "Required:\n"
        << "  -n <note>   Target MIDI note number (0–127)\n"
        << "  -o <dir>    Output directory (created if it doesn't exist)\n"
        << "              (not required with -a)\n\n"
        << "Options:\n"
        << "  -m <mode>   Processing mode (default: shift)\n"
        << "                -m shift  – single pitch correction across the whole file\n"
        << "                            suited for short/staccato samples\n"
        << "                -m drift  – per-segment correction to fix pitch drift\n"
        << "                            suited for sustained/legato samples\n"
        << "  -a          Analyse the first input file and print per-segment pitch\n"
        << "              detection details.  No output files are written.\n"
        << "              Useful for diagnosing detection quality.\n"
        << "  -c <cents>  Minimum correction threshold in cents (default: 5).\n"
        << "              Files detected within this many cents of target are\n"
        << "              copied unchanged rather than corrected.  Use 0 to\n"
        << "              always attempt correction.\n"
        << "  -t <val>    YIN confidence threshold (default: 0.15, range 0.05–0.30)\n"
        << "              Lower values are stricter (fewer false positives).\n"
        << "  -h          Show this help\n\n"
        << "Notes:\n"
        << "  Pitch analysis is performed on the first input file only.\n"
        << "  Identical corrections are then applied to every input file.\n"
        << "  This ensures multi-mic recordings stay perfectly in sync.\n"
        << "  Original files are never modified.\n\n"
        << "Examples:\n"
        << "  " << prog << " -n 60 -o tuned/ sample.wav\n"
        << "  " << prog << " -n 60 -m drift -o tuned/ close.wav room.wav\n"
        << "  " << prog << " -n 60 -a sample.wav\n";
}

static double midi_to_freq(int note)
{
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

static double freq_to_semitones(double from_hz, double to_hz)
{
    return 12.0 * std::log2(to_hz / from_hz);
}

int main(int argc, char* argv[])
{
    int         midi_note          = -1;
    std::string output_dir;
    std::string mode               = "shift";
    float       yin_threshold      = 0.15f;
    double      min_correction_cents = 5.0;
    bool        analyze_mode       = false;

    int opt;
    while ((opt = getopt(argc, argv, "n:o:m:t:c:ah")) != -1) {
        switch (opt) {
            case 'n':
                try { midi_note = std::stoi(optarg); }
                catch (...) {
                    std::cerr << "Error: invalid MIDI note '" << optarg << "'\n";
                    return 1;
                }
                break;
            case 'o': output_dir    = optarg; break;
            case 'm': mode          = optarg; break;
            case 't':
                try { yin_threshold = std::stof(optarg); }
                catch (...) {
                    std::cerr << "Error: invalid threshold '" << optarg << "'\n";
                    return 1;
                }
                break;
            case 'c':
                try { min_correction_cents = std::stod(optarg); }
                catch (...) {
                    std::cerr << "Error: invalid cents value '" << optarg << "'\n";
                    return 1;
                }
                break;
            case 'a': analyze_mode   = true;  break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    // --- Validate arguments ---
    if (midi_note < 0 || midi_note > 127) {
        std::cerr << "Error: MIDI note must be 0–127 (got "
                  << midi_note << ")\n\n";
        print_usage(argv[0]);
        return 1;
    }
    if (output_dir.empty() && !analyze_mode) {
        std::cerr << "Error: output directory (-o) is required\n\n";
        print_usage(argv[0]);
        return 1;
    }
    if (mode != "shift" && mode != "drift") {
        std::cerr << "Error: mode must be 'shift' or 'drift'\n\n";
        print_usage(argv[0]);
        return 1;
    }
    if (yin_threshold < 0.01f || yin_threshold > 0.5f) {
        std::cerr << "Error: threshold should be in the range 0.01–0.50\n";
        return 1;
    }

    std::vector<std::string> input_files;
    for (int i = optind; i < argc; ++i)
        input_files.emplace_back(argv[i]);

    if (input_files.empty()) {
        std::cerr << "Error: no input files specified\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // --- Create output directory (not needed for analyze mode) ---
    if (!analyze_mode) {
        try {
            std::filesystem::create_directories(output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Error creating output directory '" << output_dir
                      << "': " << e.what() << "\n";
            return 1;
        }
    }

    double target_freq = midi_to_freq(midi_note);

    if (analyze_mode)
        std::cout << "Analyzing: " << input_files[0] << "\n";
    else {
        std::cout << "Target : MIDI " << midi_note
                  << "  (" << target_freq << " Hz)\n";
        std::cout << "Mode   : " << mode << "\n";
        std::cout << "Files  : " << input_files.size() << "\n\n";
        std::cout << "Analysing: " << input_files[0] << "\n";
    }

    // --- Load the reference file (always the first) ---
    std::vector<float> ref_samples;
    int ref_sr, ref_ch, ref_fmt;
    long long ref_frames;

    if (!load_audio(input_files[0], ref_samples,
                    ref_sr, ref_ch, ref_frames, ref_fmt))
        return 1;

    std::cout << "  " << ref_sr << " Hz, " << ref_ch
              << " ch, " << static_cast<double>(ref_frames) / ref_sr
              << " s\n";

    // --- Analyze mode: print diagnostics then show predicted output pitch ---
    if (analyze_mode) {
        std::cout << "  Target : MIDI " << midi_note
                  << "  (" << target_freq << " Hz)\n"
                  << "  YIN threshold : " << yin_threshold << "\n";
        analyze_audio(ref_samples, ref_ch, ref_sr, target_freq, yin_threshold);

        // Compute the corrections that would actually be applied, apply them
        // to an in-memory copy, then re-detect pitch so the user can verify
        // the output with a tuner without needing a second run.
        std::vector<SegmentCorrection> prev_corr;
        double prev_global = 0.0;

        if (mode == "shift") {
            double det = detect_overall_pitch(ref_samples, ref_ch, ref_sr,
                                              target_freq, yin_threshold);
            if (det > 0.0) {
                prev_global = det;
                double ratio  = det / target_freq;
                long long out_fr = std::llround(
                    static_cast<double>(ref_frames) * ratio);
                prev_corr.push_back({ 0LL, ref_frames, out_fr, ratio });
            }
        } else { // drift
            prev_corr = compute_drift_corrections(
                ref_samples, ref_ch, ref_sr, target_freq, yin_threshold,
                &prev_global);
        }

        std::cout << "--- Predicted output (mode: " << mode << ") ---\n";

        if (prev_global <= 0.0 || prev_corr.empty()) {
            std::cout << "  Detection failed – cannot predict output pitch.\n\n";
            return 0;
        }

        double prev_cents    = 1200.0 * std::log2(prev_global / target_freq);
        double prev_deviation = std::abs(prev_cents);

        std::cout << "  Input  : " << prev_global << " Hz  ("
                  << (prev_cents >= 0 ? "+" : "") << prev_cents << " cents)\n";

        if (min_correction_cents > 0.0 && prev_deviation < min_correction_cents) {
            std::cout << "  Action : copy unchanged  (deviation "
                      << prev_deviation << "c < "
                      << min_correction_cents << "c threshold)\n";
            std::cout << "  Output : " << prev_global << " Hz  (unchanged)\n\n";
            return 0;
        }

        std::cout << "  Action : correct\n";
        std::vector<float> prev_out =
            apply_corrections(ref_samples, ref_ch, prev_corr);
        double out_pitch = detect_overall_pitch(prev_out, ref_ch, ref_sr,
                                                target_freq, yin_threshold);
        if (out_pitch > 0.0) {
            double out_cents = 1200.0 * std::log2(out_pitch / target_freq);
            std::cout << "  Output : " << out_pitch << " Hz  ("
                      << (out_cents >= 0 ? "+" : "") << out_cents
                      << " cents)\n\n";
        } else {
            std::cout << "  Output : pitch detection failed on corrected audio\n\n";
        }
        return 0;
    }

    // Below this threshold the detected deviation is within the pitch-detection
    // noise floor; applying the correction risks making the output less accurate
    // than the original.  Files within this range are copied unchanged.

    // --- Compute corrections from the reference file ---
    std::vector<SegmentCorrection> corrections;
    double detected_global = 0.0; // filled by both branches

    if (mode == "shift") {
        double detected = detect_overall_pitch(ref_samples, ref_ch, ref_sr, target_freq, yin_threshold);
        if (detected <= 0.0) {
            std::cerr << "Error: could not detect pitch in '"
                      << input_files[0] << "'\n";
            return 1;
        }
        detected_global = detected;
        double semitones = freq_to_semitones(detected, target_freq);
        std::cout << "  Detected : " << detected << " Hz\n";
        std::cout << "  Target   : " << target_freq << " Hz\n";
        std::cout << "  Shift    : " << (semitones >= 0 ? "+" : "")
                  << semitones << " semitones\n";

        // Single correction covering the whole file
        double ratio     = detected / target_freq; // src_ratio for libsamplerate
        long long out_fr = std::llround(static_cast<double>(ref_frames) * ratio);
        corrections.push_back({ 0LL, ref_frames, out_fr, ratio });

    } else { // drift
        corrections = compute_drift_corrections(
            ref_samples, ref_ch, ref_sr, target_freq, yin_threshold,
            &detected_global);
        if (corrections.empty()) {
            std::cerr << "Error: drift analysis failed for '"
                      << input_files[0] << "'\n";
            return 1;
        }

        // Print a brief summary of global shift + drift range
        double min_r = corrections[0].src_ratio;
        double max_r = corrections[0].src_ratio;
        for (const auto& c : corrections) {
            min_r = std::min(min_r, c.src_ratio);
            max_r = std::max(max_r, c.src_ratio);
        }
        double global_shift_cents = 1200.0 * std::log2(detected_global / target_freq);
        double drift_cents = 1200.0 * std::log2(max_r / min_r);
        std::cout << "  Detected : " << detected_global << " Hz\n";
        std::cout << "  Shift    : " << (global_shift_cents >= 0 ? "+" : "")
                  << global_shift_cents << " cents  (applied to all "
                  << corrections.size() << " segments)\n";
        std::cout << "  Drift    : " << drift_cents << " cents peak-to-peak\n";
    }

    // --- Guard: skip correction if the detected deviation is too small ---
    // If the overall pitch is already within kMinCorrectionCents of target,
    // the detection uncertainty is comparable to the deviation itself and
    // correcting could make the output worse than the input.
    double global_cents = std::abs(1200.0 * std::log2(detected_global / target_freq));
    if (min_correction_cents > 0.0 && global_cents < min_correction_cents) {
        std::cout << "  Within " << min_correction_cents
                  << "-cent threshold (" << global_cents
                  << " cents) – already in tune, copying originals.\n\n";

        int errors = 0;
        for (const auto& in_path : input_files) {
            std::string out_path = output_dir + "/" +
                                   std::filesystem::path(in_path).filename().string();
            std::error_code ec;
            if (std::filesystem::equivalent(in_path, out_path, ec)) {
                std::cerr << "Error: input and output paths are the same for '"
                          << in_path << "' – choose a different output directory\n";
                ++errors;
                continue;
            }
            std::cout << "Copying : " << out_path << "\n";
            std::filesystem::copy_file(in_path, out_path,
                std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                std::cerr << "Error copying '" << in_path << "': "
                          << ec.message() << "\n";
                ++errors;
            }
        }
        std::cout << "\nDone" << (errors ? " (with errors)" : "") << ".\n";
        return errors ? 1 : 0;
    }

    std::cout << "\n";

    // --- Apply corrections to every input file ---
    int errors = 0;
    for (size_t fi = 0; fi < input_files.size(); ++fi) {
        const std::string& in_path = input_files[fi];

        // Build output path: <output_dir>/<basename>
        std::string out_path = output_dir + "/" +
                               std::filesystem::path(in_path).filename().string();

        // Guard: refuse to overwrite the source
        std::error_code ec;
        if (std::filesystem::equivalent(in_path, out_path, ec)) {
            std::cerr << "Error: input and output paths are the same for '"
                      << in_path << "' – choose a different output directory\n";
            ++errors;
            continue;
        }

        std::vector<float> src_samples;
        int src_sr, src_ch, src_fmt;
        long long src_frames;

        if (fi == 0) {
            // Reuse the already-loaded reference
            src_samples = std::move(ref_samples);
            src_sr  = ref_sr;
            src_ch  = ref_ch;
            src_fmt = ref_fmt;
            src_frames = ref_frames;
        } else {
            if (!load_audio(in_path, src_samples,
                            src_sr, src_ch, src_frames, src_fmt)) {
                ++errors;
                continue;
            }
        }

        std::vector<float> output = apply_corrections(src_samples, src_ch, corrections);

        std::cout << "Writing : " << out_path << "\n";
        if (!save_audio(out_path, output, src_sr, src_ch, src_fmt))
            ++errors;
    }

    std::cout << "\nDone" << (errors ? " (with errors)" : "") << ".\n";
    return errors ? 1 : 0;
}
