#pragma once

#include <vector>

// YIN fundamental frequency estimator.
// Reference: De Cheveigné & Kawahara (2002)
// "YIN, a fundamental frequency estimator for speech and music"
class YIN {
public:
    // sample_rate : audio sample rate in Hz
    // window_size : analysis window in samples (must be even)
    //               minimum detectable freq = sample_rate / (window_size / 2)
    // threshold   : aperiodicity acceptance threshold (0.05–0.25)
    //               lower = stricter, fewer false positives
    YIN(int sample_rate, int window_size = 4096, float threshold = 0.15f);

    // Detect the fundamental frequency in `buffer`.
    // `size` must be >= window_size().
    // Returns frequency in Hz, or 0.0f if no pitch is found.
    // Optionally fills `confidence` with a 0–1 reliability score.
    float detect(const float* buffer, int size, float* confidence = nullptr) const;

    int window_size() const { return window_size_; }

private:
    int   sample_rate_;
    int   window_size_;
    float threshold_;
};
