/*
 * S4FIFO LightGBM Ensemble Predictor
 * 
 * LightGBM ensemble model (5 models, 50 trees each) for predicting optimal
 * S4FIFO configuration parameters based on cache workload features.
 * 
 * This is generated from train_xgb_18class_lite.py and exported via m2cgen.
 * Model accuracy: ~47% top-1, Mean improvement over FIFO: 15.34%
 * 
 * NOTE: This header should be included AFTER S4FIFOFeatureVector and
 * S4FIFOPredictedParams are defined (e.g., by MMS4FIFO.h).
 */

#pragma once

#include <cmath>
#include <cstddef>  // for size_t

// Include the ensemble model (C code, now header-only)
extern "C" {
#include "s4fifo_model/S4FIFOEnsemble.h"
}

// NOTE: This file is included from within namespace facebook::cachelib in MMS4FIFO.h
// Do NOT add a namespace declaration here.

// The 18 selected S3FIFO/S4FIFO configurations
// Format: (s_param, m_param, t_param, g_param, k_param)
// s_param -> tinySizePercent: small queue size as fraction (0.05-0.9) * 100 = 5-90%
// m_param -> moveToMainThreshold: frequency threshold for small->main (1-2)
// t_param -> ghostToMainThreshold: ghost->main threshold (0-1)
// g_param -> ghostSizePercent: ghost size as fraction (0.9-6.0) * 100 = 90-600%
// k_param -> smallSkipRatio: skip ratio for frequency increment (always 0.25)

struct S4FIFOConfigEntry {
    double s;  // tinySizePercent (fraction, multiply by 100)
    int m;     // moveToMainThreshold  
    int t;     // ghostToMainThreshold
    double g;  // ghostSizePercent (fraction, multiply by 100)
    double k;  // smallSkipRatio
};


// // Raw input features (from cache statistics)
// typedef struct {
//     // Histogram bins (20 bins each for ghost, main, small)
//     double hist_ghost[20];
//     double hist_main[20];
//     double hist_small[20];
    
//     // Entropy
//     double H_g, H_m, H_s;
    
//     // Hit counts
//     double hits_ghost, hits_main, hits_small, total_hits;
    
//     // Request stats
//     double rho_onehit, rho_unique, total_reqs;
    
//     // Cache size (log scale) - from actual cache size
//     double log_C;  // log(cache_size)
// } RawFeatures;


constexpr S4FIFOConfigEntry kS4FIFOConfigs[18] = {
    {0.20, 1, 0, 3.0, 0.25},  // Class 0
    {0.05, 1, 0, 0.9, 0.25},  // Class 1
    {0.50, 1, 0, 0.9, 0.25},  // Class 2
    {0.20, 1, 0, 0.9, 0.25},  // Class 3
    {0.05, 2, 0, 6.0, 0.25},  // Class 4
    {0.10, 2, 1, 3.0, 0.25},  // Class 5
    {0.30, 2, 0, 3.0, 0.25},  // Class 6
    {0.05, 2, 0, 3.0, 0.25},  // Class 7
    {0.10, 2, 0, 0.9, 0.25},  // Class 8
    {0.70, 1, 1, 0.9, 0.25},  // Class 9
    {0.20, 1, 1, 0.9, 0.25},  // Class 10
    {0.05, 1, 1, 0.9, 0.25},  // Class 11
    {0.30, 1, 0, 6.0, 0.25},  // Class 12
    {0.20, 2, 0, 0.9, 0.25},  // Class 13
    {0.90, 2, 0, 3.0, 0.25},  // Class 14
    {0.10, 2, 0, 6.0, 0.25},  // Class 15
    {0.30, 2, 1, 3.0, 0.25},  // Class 16
    {0.05, 2, 0, 0.9, 0.25},  // Class 17
};



// Feature order (alphabetical, matching Python training):
// 0: H_g, 1: H_m, 2: H_s, 3: decay_rate_small, 4: entropy_gap, 5: ghost_pressure,
// 6-25: hist_ghost_0..19, 26-45: hist_main_0..19, 46-65: hist_small_0..19,
// 66: log_C, 67: probation_efficiency, 68: ratio_estimate, 69: rho_onehit,
// 70: rho_unique, 71: scan_intensity, 72: tail_heaviness, 73: thrashing_risk, 74: total_reqs

// Prepare model input features from S4FIFOFeatureVector
// Maps the CacheLib feature vector to the 75-feature input expected by the model
// Features are in SORTED order (alphabetical) as per feature_columns.json
inline void prepareModelInput(const S4FIFOFeatureVector& fv, double* input) {
    // Initialize all to 0
    for (int i = 0; i < 75; i++) input[i] = 0.0;
    
    // Get log_C from feature vector (log10 of cache capacity in objects)
    // fv.logCacheCapacity is already log10, matching Python training
    double log_C = fv.logCacheCapacity;
    
    // === Derived features ===
    
    // probation_efficiency: hits_small / (hits_main + 1e-6)
    double probation_efficiency = static_cast<double>(fv.hitsSmall) / (fv.hitsMain + 1e-6);
    
    // ghost_pressure: hits_ghost / (total_hits + hits_ghost + 1e-6)
    double total_hits_with_ghost = static_cast<double>(fv.totalHits + fv.hitsGhost);
    double ghost_pressure = static_cast<double>(fv.hitsGhost) / (total_hits_with_ghost + 1e-6);
    
    // entropy_gap: H_m - H_s
    double entropy_gap = fv.hitRatioMain - fv.hitRatioSmall;
    
    // decay_rate_small: hist_small[0] - hist_small[1]
    double decay_rate_small = fv.histSmall[0] - fv.histSmall[1];
    
    // tail_heaviness: sum(hist_main[10..19])
    double tail_heaviness = 0.0;
    for (int i = 10; i < 20; i++) tail_heaviness += fv.histMain[i];
    
    // === WSS-Estimated Ratio ===
    // ratio_estimate = cache_size / working_set_size
    // cache_size = exp(log_C)
    // working_set_size = total_reqs * rho_unique
    double cache_size = std::exp(log_C);
    double working_set_size = static_cast<double>(fv.totalRequests) * fv.uniqueRatio;
    double ratio_estimate = cache_size / (working_set_size + 1e-6);
    
    // Clip to reasonable range [0.0001, 1.0]
    if (ratio_estimate < 0.0001) ratio_estimate = 0.0001;
    if (ratio_estimate > 1.0) ratio_estimate = 1.0;
    
    // thrashing_risk = rho_unique / (ratio_estimate * 100 + 1e-6)
    double thrashing_risk = fv.uniqueRatio / (ratio_estimate * 100.0 + 1e-6);
    
    // scan_intensity = rho_onehit * (1 - ratio_estimate)
    double scan_intensity = fv.oneHitRatio * (1.0 - ratio_estimate);
    
    // === Fill input array (alphabetical order) ===
    input[0] = fv.hitRatioGhost;            // H_g
    input[1] = fv.hitRatioMain;             // H_m
    input[2] = fv.hitRatioSmall;            // H_s
    input[3] = decay_rate_small;            // decay_rate_small
    input[4] = entropy_gap;                 // entropy_gap
    input[5] = ghost_pressure;              // ghost_pressure
    
    // IMPORTANT: Python sorts feature names alphabetically, so histogram indices are:
    // 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9
    // This is lexicographic (string) order, not numeric order!
    static const int HIST_ORDER[20] = {0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // hist_ghost_0, 1, 10, 11, ... 19, 2, 3, ... 9 (indices 6-25)
    for (int i = 0; i < 20; i++) {
        input[6 + i] = fv.histGhost[HIST_ORDER[i]];
    }
    
    // hist_main_0, 1, 10, 11, ... 19, 2, 3, ... 9 (indices 26-45)
    for (int i = 0; i < 20; i++) {
        input[26 + i] = fv.histMain[HIST_ORDER[i]];
    }
    
    // hist_small_0, 1, 10, 11, ... 19, 2, 3, ... 9 (indices 46-65)
    for (int i = 0; i < 20; i++) {
        input[46 + i] = fv.histSmall[HIST_ORDER[i]];
    }
    
    input[66] = log_C;                      // log_C
    input[67] = probation_efficiency;       // probation_efficiency
    input[68] = ratio_estimate;             // ratio_estimate
    input[69] = fv.oneHitRatio;             // rho_onehit
    input[70] = fv.uniqueRatio;             // rho_unique
    input[71] = scan_intensity;             // scan_intensity
    input[72] = tail_heaviness;             // tail_heaviness
    input[73] = thrashing_risk;             // thrashing_risk
    input[74] = static_cast<double>(fv.totalRequests);  // total_reqs
}

// LightGBM prediction function to be used as S4FIFOPredictionCallback
// Uses features.logCacheCapacity for log_C (automatically populated from cache size)
inline S4FIFOPredictedParams lightGBMPredict(const S4FIFOFeatureVector& features) {
    S4FIFOPredictedParams params;
    
    // Skip prediction if not enough data (need some hits to be meaningful)
    if (features.totalRequests < 10000 || features.totalHits < 100) {
        // Return "no change" params
        return params;
    }
    
    // Prepare input features (75 features including log_C and ratio_estimate)
    double input[75];
    prepareModelInput(features, input);
    
    // Run ensemble prediction
    int classId = ensemble_predict(input);
    
    // Validate class ID
    if (classId < 0 || classId >= 18) {
        // Return "no change" params on error
        return params;
    }
    
    // Get the predicted config
    const S4FIFOConfigEntry& config = kS4FIFOConfigs[classId];
    
    // Map config to CacheLib parameters
    // s_param (0.05-0.9) -> tinySizePercent (5-90%)
    params.tinySizePercent = static_cast<size_t>(config.s * 100);
    
    // m_param (1-2) -> moveToMainThreshold
    params.moveToMainThreshold = config.m;
    
    // t_param (0-1) -> ghostToMainThreshold
    params.ghostToMainThreshold = config.t;
    
    // g_param (0.9-6.0) -> ghostSizePercent (90-600%)
    size_t ghostPercent = static_cast<size_t>(config.g * 100);
    if (ghostPercent > 600) ghostPercent = 600;  // Cap at 600%
    if (ghostPercent < 10) ghostPercent = 10;    // Min 10%
    params.ghostSizePercent = ghostPercent;
    
    // k_param (always 0.25) -> smallSkipRatio
    params.smallSkipRatio = config.k;
    
    return params;
}

// namespace facebook::cachelib is provided by the including file (MMS4FIFO.h)
