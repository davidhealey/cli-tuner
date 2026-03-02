#include "yin.h"

#include <cmath>
#include <vector>
#include <algorithm>

YIN::YIN(int sample_rate, int window_size, float threshold)
    : sample_rate_(sample_rate)
    , window_size_(window_size)
    , threshold_(threshold)
{}

float YIN::detect(const float* buffer, int size, float* confidence) const
{
    if (size < window_size_) {
        if (confidence) *confidence = 0.0f;
        return 0.0f;
    }

    const int W = window_size_ / 2; // half-window = tau search range

    // --- Step 1: difference function ---
    // d[tau] = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
    std::vector<float> d(W, 0.0f);
    for (int tau = 1; tau < W; ++tau) {
        for (int j = 0; j < W; ++j) {
            float diff = buffer[j] - buffer[j + tau];
            d[tau] += diff * diff;
        }
    }

    // --- Step 2: cumulative mean normalised difference function ---
    // d'[0]   = 1
    // d'[tau] = d[tau] * tau / sum_{j=1}^{tau} d[j]
    std::vector<float> dp(W);
    dp[0] = 1.0f;
    float running_sum = 0.0f;
    for (int tau = 1; tau < W; ++tau) {
        running_sum += d[tau];
        dp[tau] = (running_sum > 1e-8f) ? (d[tau] * tau / running_sum) : 0.0f;
    }

    // --- Step 3: absolute threshold ---
    // Find the first local minimum of dp[] that is below the threshold.
    int tau_min = -1;
    for (int tau = 2; tau < W - 1; ++tau) {
        if (dp[tau] < threshold_) {
            // Walk downhill to the true local minimum
            while (tau + 1 < W - 1 && dp[tau + 1] <= dp[tau])
                ++tau;
            tau_min = tau;
            break;
        }
    }

    // If no threshold crossing, fall back to the global minimum (lower confidence)
    bool is_reliable = (tau_min >= 0);
    if (!is_reliable) {
        tau_min = 2;
        for (int tau = 3; tau < W - 1; ++tau)
            if (dp[tau] < dp[tau_min]) tau_min = tau;
    }

    if (tau_min < 2) {
        if (confidence) *confidence = 0.0f;
        return 0.0f;
    }

    // --- Step 4: parabolic interpolation for sub-sample accuracy ---
    float tau_frac = static_cast<float>(tau_min);
    if (tau_min > 1 && tau_min < W - 1) {
        float s0 = dp[tau_min - 1];
        float s1 = dp[tau_min];
        float s2 = dp[tau_min + 1];
        float denom = 2.0f * (2.0f * s1 - s2 - s0);
        if (std::abs(denom) > 1e-8f) {
            float adjust = (s2 - s0) / denom;
            tau_frac = tau_min + std::max(-0.5f, std::min(0.5f, adjust));
        }
    }

    // --- Confidence ---
    if (confidence) {
        float dp_val = (tau_min < W) ? dp[tau_min] : 1.0f;
        float conf = 1.0f - std::max(0.0f, std::min(1.0f, dp_val));
        *confidence = is_reliable ? conf : conf * 0.5f;
    }

    return (tau_frac >= 2.0f) ? (sample_rate_ / tau_frac) : 0.0f;
}
