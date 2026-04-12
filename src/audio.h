#pragma once

#include <string>
#include <vector>

// Describes a single pitch-correction segment.
// The src_ratio follows libsamplerate convention:
//   src_ratio = output_frames / input_frames
//   < 1.0  →  fewer output samples  →  plays faster  →  pitch goes UP
//   > 1.0  →  more  output samples  →  plays slower  →  pitch goes DOWN
struct SegmentCorrection {
    long long input_offset;   // Frame offset into the source audio
    long long input_frames;   // Number of input frames in this segment
    long long output_frames;  // Expected output frames after resampling
    double    src_ratio;      // detected_freq / target_freq
};

// ---------------------------------------------------------------------------
// Audio I/O
// ---------------------------------------------------------------------------

// Load an audio file into interleaved float samples.
// `format` is filled with the libsndfile format code for later writing.
// Returns false on error.
bool load_audio(const std::string& path,
                std::vector<float>& samples,
                int&       sample_rate,
                int&       channels,
                long long& total_frames,
                int&       format);

// Save interleaved float samples to an audio file.
// `format` should come from load_audio() on the original file.
// Returns false on error.
bool save_audio(const std::string& path,
                const std::vector<float>& samples,
                int sample_rate,
                int channels,
                int format);

// ---------------------------------------------------------------------------
// Pitch analysis
// ---------------------------------------------------------------------------

// Detect the overall fundamental frequency of an audio recording.
// Uses YIN on multiple windows and returns the median (robust against outliers).
// Returns frequency in Hz, or 0 on failure.
double detect_overall_pitch(const std::vector<float>& samples,
                            int       channels,
                            int       sample_rate,
                            double    target_freq,
                            float     yin_threshold);

// Compute per-segment drift corrections to retune audio toward target_freq.
// Analysis is done on the first channel of the interleaved buffer.
// Returns a list of SegmentCorrection structs that cover the entire file.
std::vector<SegmentCorrection> compute_drift_corrections(
    const std::vector<float>& samples,
    int    channels,
    int    sample_rate,
    double target_freq,
    float  yin_threshold);

// ---------------------------------------------------------------------------
// Correction application
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Diagnostic analysis
// ---------------------------------------------------------------------------

// Print a detailed per-segment pitch analysis to stdout without modifying any
// audio.  Useful for diagnosing detection quality before committing to drift
// mode.  Prints:
//   • global detected pitch and deviation from target
//   • per-segment raw and smoothed frequencies
//   • whether each segment would receive a per-segment or global-only correction
void analyze_audio(const std::vector<float>& samples,
                   int    channels,
                   int    sample_rate,
                   double target_freq,
                   float  yin_threshold);

// Apply a list of corrections to audio, returning the pitch-corrected output.
//   shift mode : pass a single SegmentCorrection covering the whole file.
//   drift mode : pass the list from compute_drift_corrections().
// If the source file is shorter than what the corrections describe, the
// remaining corrections are silently skipped.
std::vector<float> apply_corrections(const std::vector<float>& samples,
                                     int channels,
                                     const std::vector<SegmentCorrection>& corrections);
