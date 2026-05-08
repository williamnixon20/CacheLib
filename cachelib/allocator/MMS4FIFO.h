/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <folly/Format.h>
#include <folly/Math.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/synchronization/DistributedMutex.h>

#pragma GCC diagnostic pop

#include "cachelib/allocator/Cache.h"
#include "cachelib/allocator/CacheStats.h"
#include "cachelib/allocator/Util.h"
#include "cachelib/allocator/datastruct/MultiDList.h"
#include "cachelib/allocator/memory/serialize/gen-cpp2/objects_types.h"
#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/FIFOConcurrentHashSet.h"
#include "cachelib/common/FIFOHashSet.h"

namespace facebook::cachelib {

// ============================================================================
// S4FIFO Feature Collection Structures
// ============================================================================

// Maximum number of buckets for hit position histograms
constexpr int32_t kS4FIFOMaxBuckets = 64;
constexpr int32_t kS4FIFODefaultBuckets = 20;
constexpr size_t kS4FIFOMinTrackedLruSize = 1000;
inline constexpr bool shouldLog{false};

// Feature vector containing collected statistics
struct S4FIFOFeatureVector {
  int32_t numBuckets{kS4FIFODefaultBuckets};
  double logCacheCapacity{0.0};

  // Hit ratios by queue (spatial distribution)
  double hitRatioSmall{0.0}; // H_s: fraction of hits in Small FIFO
  double hitRatioMain{0.0};  // H_m: fraction of hits in Main FIFO
  double hitRatioGhost{0.0}; // H_g: ghost hit ratio (frequency signal)

  // Workload proxies
  double uniqueRatio{0.0}; // ρ_unique: unique_objs / total_requests
  double oneHitRatio{0.0}; // ρ_onehit: one-hit wonders / total_unique

  // Absolute values
  int64_t totalRequests{0};
  int64_t totalHits{0};
  int64_t totalMisses{0};
  int64_t hitsSmall{0};
  int64_t hitsMain{0};
  int64_t hitsGhost{0};

  // Hit position histograms (normalized)
  double histSmall[kS4FIFOMaxBuckets]{};
  double histMain[kS4FIFOMaxBuckets]{};
  double histGhost[kS4FIFOMaxBuckets]{};
};

// Predicted parameters returned by the prediction function
// Use sentinel values to indicate "no change":
//   SIZE_MAX for size_t, -1 for int, -1.0 for double
struct S4FIFOPredictedParams {
  // SIZE_MAX means "do not change this parameter"
  size_t tinySizePercent{SIZE_MAX};  // Tiny queue size percentage
  size_t ghostSizePercent{SIZE_MAX}; // Ghost queue size percentage
  int moveToMainThreshold{-1};       // Frequency threshold for small->main
  double smallSkipRatio{-1.0};       // Skip ratio for frequency increment
  int ghostToMainThreshold{-1};      // Frequency threshold for ghost->main

  // Check if any parameter should be updated
  bool hasAnyChange() const {
    return tinySizePercent != SIZE_MAX || ghostSizePercent != SIZE_MAX ||
           moveToMainThreshold != -1 || smallSkipRatio >= 0.0 ||
           ghostToMainThreshold != -1;
  }
};

// Prediction callback function type
// Takes feature vector, returns predicted parameters
using S4FIFOPredictionCallback =
    std::function<S4FIFOPredictedParams(const S4FIFOFeatureVector&)>;

// Include the LightGBM predictor after type definitions
#include "cachelib/allocator/S4FIFOLightGBMPredictor.h"

// Bucketed hit position tracker for O(1) hit position recording
struct S4FIFOHitPosTracker {
  int32_t numBuckets{kS4FIFODefaultBuckets};
  int64_t bucketSize{1};

  int64_t hitCounts[kS4FIFOMaxBuckets]{};
  int64_t totalHits{0};

  // For ghost queue: track middle removals (holes)
  bool trackMiddleRemoval{false};
  int64_t removalCounters[kS4FIFOMaxBuckets]{};
  int64_t totalRemovals{0};
  int64_t currentBucket{0};

  bool haveWarmedUp = false;

  void init(int64_t expectedMaxPos, int32_t buckets, bool trackRemoval) {
    if (buckets <= 0)
      buckets = kS4FIFODefaultBuckets;
    if (buckets > kS4FIFOMaxBuckets)
      buckets = kS4FIFOMaxBuckets;
    numBuckets = buckets;
    auto newBucketSize = (expectedMaxPos + buckets - 1) / buckets;
    if (newBucketSize > 1) {
      bucketSize = newBucketSize;
    } else {
      bucketSize = 1;
    }
    trackMiddleRemoval = trackRemoval;

    if (shouldLog) {
      printf(
          "S4FIFOHitPosTracker initialized: expectedMaxPos=%ld buckets=%d bucketSize=%ld trackRemoval=%d\n",
          expectedMaxPos, numBuckets, bucketSize, trackMiddleRemoval);
    }
  }

  void setWarmedUp() {
    if (shouldLog) {
      printf("S4FIFOHitPosTracker warmed up\n");
    }
    haveWarmedUp = true;
  }

  // facebook::cachelib::MMS4FIFO::Container<facebook::cachelib::CacheItem<facebook::cachelib::S4FIFOCacheTrait>, &facebook::cachelib::CacheItem<facebook::cachelib::S4FIFOCacheTrait>::mmHook_>::remove(facebook::cachelib::MMS4FIFO::Container<facebook::cachelib::CacheItem<facebook::cachelib::S4FIFOCacheTrait>, &facebook::cachelib::CacheItem<facebook::cachelib::S4FIFOCacheTrait>::mmHook_>::LockedIterator&)
  // Insertions done
  int64_t recordInsert(int64_t insertCounter) {
    if (!haveWarmedUp)
      return 0;

    if (bucketSize == 0) {
      return 0;
    }
    int64_t newBucket = (insertCounter / bucketSize) % numBuckets;
    if (trackMiddleRemoval && newBucket != currentBucket) {
      int64_t b = (currentBucket + 1) % numBuckets;
      while (b != newBucket) {
        removalCounters[b] = 0;
        b = (b + 1) % numBuckets;
      }
      removalCounters[newBucket] = 0;
    }
    currentBucket = newBucket;
    return newBucket;
  }

  // Maybe inaccurate for now
  void recordRemoval() {
    if (!haveWarmedUp)
      return;
    if (!trackMiddleRemoval)
      return;
    removalCounters[currentBucket]++;
    totalRemovals++;
  }

  // Private dont need to use.
  int64_t estimateHoles(int64_t insertBucket) const {
    if (!haveWarmedUp)
      return 0;
    if (!trackMiddleRemoval)
      return 0;
    int64_t holes = 0;
    int64_t bucket = insertBucket;
    while (bucket != currentBucket) {
      holes += removalCounters[bucket];
      bucket = (bucket + 1) % numBuckets;
    }
    holes += removalCounters[currentBucket];
    return holes;
  }

  // Hits done
  // Need insert bucket for ghost
  void recordHit(int64_t insertTime,
                 int64_t insertBucket,
                 int64_t currentCounter) {
    if (!haveWarmedUp)
      return;
    if (bucketSize == 0) {
      return;
    }
    int64_t rawPosition = currentCounter - insertTime;
    int64_t holes = estimateHoles(insertBucket);
    int64_t adjustedPosition = rawPosition - holes;
    if (adjustedPosition < 0)
      adjustedPosition = 0;
    int64_t posBucket = adjustedPosition / bucketSize;
    if (posBucket >= numBuckets) {
      posBucket = numBuckets - 1;
    }
    hitCounts[posBucket]++;
    totalHits++;
  }

  void getHistogram(double* out) const {
    if (totalHits == 0) {
      memset(out, 0, numBuckets * sizeof(double));
      return;
    }
    double invTotal = 1.0 / static_cast<double>(totalHits);
    for (int i = 0; i < numBuckets; i++) {
      out[i] = static_cast<double>(hitCounts[i]) * invTotal;
    }
  }

  // getHistogramString()
  std::string getHistogramString() const {
    std::string result = "[";
    for (int i = 0; i < numBuckets; i++) {
      if (i > 0) {
        result += ", ";
      }
      result += folly::to<std::string>(hitCounts[i]);
    }
    result += "]";
    return result;
  }
  

  void reset() {
    memset(hitCounts, 0, sizeof(hitCounts));
    totalHits = 0;
    memset(removalCounters, 0, sizeof(removalCounters));
    totalRemovals = 0;
  }
};

// Feature collector that continuously gathers statistics
struct S4FIFOFeatureCollector {
  int64_t cacheCapacity{0};
  int32_t numBuckets{kS4FIFODefaultBuckets};

  // Per-queue insertion counters
  int64_t smallInsertCounter{0};
  int64_t mainInsertCounter{0};
  int64_t ghostInsertCounter{0};

  // Hit position trackers
  S4FIFOHitPosTracker smallTracker;
  S4FIFOHitPosTracker mainTracker;
  S4FIFOHitPosTracker ghostTracker;

  // Hit counters
  int64_t totalHitsSmall{0};
  int64_t totalHitsMain{0};
  int64_t totalHitsGhost{0};
  int64_t totalRequests{0};
  int64_t totalUnique{0};
  int64_t totalMisses{0};
  int64_t oneHitCount{0};

  int64_t capacity{0};
  int64_t sizeSmall{0};
  int64_t sizeMain{0};

  void init(int64_t capacity,
            int64_t smallSize,
            int64_t mainSize,
            int64_t ghostSize,
            int32_t buckets) {
    memset(this, 0, sizeof(S4FIFOFeatureCollector));
    cacheCapacity = capacity;
    numBuckets = buckets > 0 ? buckets : kS4FIFODefaultBuckets;
    if (numBuckets > kS4FIFOMaxBuckets)
      numBuckets = kS4FIFOMaxBuckets;

    smallTracker.init(smallSize, numBuckets, false);
    mainTracker.init(mainSize, numBuckets, false);
    ghostTracker.init(ghostSize, numBuckets, true);
    this->capacity = capacity;
    sizeSmall = smallSize;
    sizeMain = mainSize;
  }

  void setWarmedUp() {
    smallTracker.setWarmedUp();
    mainTracker.setWarmedUp();
    ghostTracker.setWarmedUp();
  }

  void reset() {
    totalHitsSmall = 0;
    totalHitsMain = 0;
    totalHitsGhost = 0;
    totalRequests = 0;
    totalUnique = 0;
    totalMisses = 0;
    oneHitCount = 0;
    smallTracker.reset();
    mainTracker.reset();
    ghostTracker.reset();
  }

  void getFeatures(S4FIFOFeatureVector& fv) const {
    fv.numBuckets = numBuckets;
    fv.logCacheCapacity = cacheCapacity > 0
                              ? std::log10(static_cast<double>(cacheCapacity))
                              : 0.0;

    fv.totalRequests = totalRequests;
    fv.hitsSmall = totalHitsSmall;
    fv.hitsMain = totalHitsMain;
    fv.hitsGhost = totalHitsGhost;
    fv.totalHits = totalHitsSmall + totalHitsMain + totalHitsGhost;
    fv.totalMisses = totalMisses;

    if (fv.totalHits > 0) {
      fv.hitRatioSmall = static_cast<double>(totalHitsSmall) / fv.totalHits;
      fv.hitRatioMain = static_cast<double>(totalHitsMain) / fv.totalHits;
      fv.hitRatioGhost = static_cast<double>(totalHitsGhost) / fv.totalHits;
    } else {
      fv.hitRatioSmall = 0.0;
      fv.hitRatioMain = 0.0;
      fv.hitRatioGhost = 0.0;
    }

    fv.uniqueRatio = totalRequests > 0
                         ? static_cast<double>(totalUnique) / totalRequests
                         : 0.0;
    fv.oneHitRatio =
        totalUnique > 0 ? static_cast<double>(oneHitCount) / totalUnique : 0.0;

    smallTracker.getHistogram(fv.histSmall);
    mainTracker.getHistogram(fv.histMain);
    ghostTracker.getHistogram(fv.histGhost);
  }

  std::string histStringify(const double* hist) const {
    std::string result = "[";
    for (int i = 0; i < numBuckets; i++) {
      if (i > 0) {
        result += ", ";
      }
      result += folly::to<std::string>(hist[i]);
    }
    result += "]";
    return result;
  }

  // To string method
  std::string toString() const {
    S4FIFOFeatureVector fv;
    getFeatures(fv);
    return folly::sformat(
        "S4FIFO Feature Vector:\n"
        "capacity: {}\n"
        "  sizeSmall: {}\n"
        "  sizeMain: {}\n"
        "  numBuckets: {}\n"
        "  logCacheCapacity: {:.4f}\n"
        "  totalRequests: {}\n"
        "  totalHits: {}\n"
        "  totalMisses: {}\n"
        "  hitRatioSmall: {:.4f}\n"
        "  hitRatioMain: {:.4f}\n"
        "  hitRatioGhost: {:.4f}\n"
        "  uniqueRatio: {:.4f}\n"
        "  oneHitRatio: {:.4f}\n"
        " histSmall: {}\n"
        " histMain: {}\n"
        " histGhost: {}\n",
        capacity, sizeSmall, sizeMain,
        fv.numBuckets, fv.logCacheCapacity, fv.totalRequests, fv.totalHits,
        fv.totalMisses, fv.hitRatioSmall, fv.hitRatioMain, fv.hitRatioGhost,
        fv.uniqueRatio, fv.oneHitRatio, histStringify(fv.histSmall), histStringify(fv.histMain), histStringify(fv.histGhost));
  }
};

// ============================================================================
// MMS4FIFO Container Implementation
// ============================================================================

// Implements the S4-FIFO cache eviction policy with continuous feature
// collection and periodical prediction-based parameter updates.
//
// Key features:
// - Frequency-based promotion thresholds (moveToMainThreshold, etc.)
// - Continuous feature collection after warmup
// - Time-based periodical parameter updates (configurable interval in seconds)
// - Prediction callback for ML-based parameter tuning
// - Bool switch for one-time vs infinite periodic updates

class MMS4FIFO {
 public:
  // unique identifier per MMType
  static const int kId;

  // forward declaration;
  template <typename T>
  using Hook = DListHook<T>;
  using SerializationType = serialization::MMS4FIFOObject;
  using SerializationConfigType = serialization::MMS4FIFOConfig;
  using SerializationTypeContainer = serialization::MMS4FIFOCollection;

  enum LruType { Main, Tiny, NumTypes };

  // Config class for MMS4FIFO
  struct Config {
    // create from serialized config
    explicit Config(SerializationConfigType configState)
        : Config(*configState.updateOnWrite(),
                 *configState.updateOnRead(),
                 *configState.tinySizePercent(),
                 *configState.ghostSizePercent(),
                 *configState.moveToMainThreshold(),
                 *configState.smallSkipRatio(),
                 *configState.ghostToMainThreshold()) {}

    // @param udpateOnW   whether to promote the item on write
    // @param updateOnR   whether to promote the item on read
    Config(bool updateOnW, bool updateOnR)
        : Config(updateOnW, updateOnR, 10, 90, 2, 0.0, 0) {}

    // Full constructor with all S4FIFO parameters
    Config(bool updateOnW,
           bool updateOnR,
           size_t tinySizePercent,
           size_t ghostSizePercent,
           int moveToMainThreshold,
           double smallSkipRatio,
           int ghostToMainThreshold)
        : updateOnWrite(updateOnW),
          updateOnRead(updateOnR),
          tinySizePercent(tinySizePercent),
          ghostSizePercent(ghostSizePercent),
          moveToMainThreshold(moveToMainThreshold),
          smallSkipRatio(smallSkipRatio),
          ghostToMainThreshold(ghostToMainThreshold) {}

    Config(bool updateOnW,
           bool updateOnR,
           size_t tinySizePercent,
           size_t ghostSizePercent,
           int moveToMainThreshold,
           double smallSkipRatio,
           int ghostToMainThreshold,
           bool enableFeatureCollection,
           uint64_t featureUpdateIntervalSecs,
           bool enablePeriodicUpdates)
        : updateOnWrite(updateOnW),
          updateOnRead(updateOnR),
          tinySizePercent(tinySizePercent),
          ghostSizePercent(ghostSizePercent),
          moveToMainThreshold(moveToMainThreshold),
          smallSkipRatio(smallSkipRatio),
          ghostToMainThreshold(ghostToMainThreshold),
          enableFeatureCollection(enableFeatureCollection),
          featureUpdateIntervalSecs(featureUpdateIntervalSecs),
          enablePeriodicUpdates(enablePeriodicUpdates) {}

    void addExtraConfig(size_t tSize) { 
      if (shouldLog) {
        printf("Setting tailSize to %zu\n", tSize);
      }
      tailSize = tSize; 
    }
    Config() = default;
    Config(const Config& rhs) = default;
    Config(Config&& rhs) = default;

    Config& operator=(const Config& rhs) = default;
    Config& operator=(Config&& rhs) = default;

    // whether the cache needs to be updated on writes for recordAccess.
    bool updateOnWrite{false};

    // whether the cache needs to be updated on reads for recordAccess.
    bool updateOnRead{true};

    // The size of tiny cache, as a percentage of the total size.
    size_t tinySizePercent{10};

    // The size of ghost queue, as a percentage of total size
    size_t ghostSizePercent{90};

    // S4FIFO-specific: frequency threshold for small->main promotion
    // Vary parameter in cachebench. 
    // Grid search in cachebench.
    int moveToMainThreshold{1};

    // S4FIFO-specific: ratio to skip frequency increment in small queue
    double smallSkipRatio{0.0};

    // S4FIFO-specific: frequency threshold for ghost->main promotion
    int ghostToMainThreshold{0};

    // ========== Feature Collection & Prediction Settings ==========

    // Enable feature collection (default: false)
    bool enableFeatureCollection{true};

    // Time interval in seconds for periodical feature updates (default: 1440s =
    // 24min) After warmup, features are collected and prediction is called
    // every this interval
    uint64_t featureUpdateIntervalSecs{1440};
    // Interval to call model, a parameter to control.

    // If true, continuously update parameters periodically
    // If false, only update once after first interval
    bool enablePeriodicUpdates{false};

    // Number of histogram buckets for feature collection
    int32_t featureNumBuckets{kS4FIFODefaultBuckets};
    // Hack to track allocation class
    size_t tailSize{0};
  };

  // The container object which can be used to keep track of objects of type T
  template <typename T, Hook<T> T::* HookPtr>
  struct Container {
   private:
    using LruList = MultiDList<T, HookPtr>;
    using Iterator = typename LruList::DListIterator;
    using Mutex = folly::DistributedMutex;
    using LockHolder = std::unique_lock<Mutex>;
    using PtrCompressor = typename T::PtrCompressor;
    using Time = typename Hook<T>::Time;
    using CompressedPtrType = typename T::CompressedPtrType;
    using RefFlags = typename T::Flags;

   public:
    Container() {};
    // This one is used
    Container(Config c, PtrCompressor compressor)
        : lru_(LruType::NumTypes, std::move(compressor)),
          config_(std::move(c)) {
      initFeatureCollection();
      if (shouldLog_) {
        printf(
            "S4 FIFO Configs are: updateOnWrite=%d, updateOnRead=%d, "
            "tinySizePercent=%zu, ghostSizePercent=%zu, skipRatio=%f, "
            "moveToMainThreshold=%d, ghostToMainThreshold=%d, enableFeatureCollection=%d, featureUpdateIntervalSecs=%lu, enablePeriodicUpdates=%d\n",
            config_.updateOnWrite, config_.updateOnRead,
            config_.tinySizePercent, config_.ghostSizePercent,
            config_.smallSkipRatio, config_.moveToMainThreshold,
            config_.ghostToMainThreshold, config_.enableFeatureCollection,
            config_.featureUpdateIntervalSecs,
            config_.enablePeriodicUpdates);
      }
    }
    Container(serialization::MMS4FIFOObject object, PtrCompressor compressor);

    Container(const Container&) = delete;
    Container& operator=(const Container&) = delete;

    // records the information that the node was accessed.
    bool recordAccess(T& node, AccessMode mode) noexcept;

    // adds the given node into the container
    bool add(T& node) noexcept;

    // removes the node from the container
    bool remove(T& node) noexcept;

    class LockedIterator;

    void remove(LockedIterator& it) noexcept;

    bool replace(T& oldNode, T& newNode) noexcept;

    // ========== Feature Collection API ==========

    // Set the prediction callback function
    void setPredictionCallback(S4FIFOPredictionCallback callback) {
      predictionCallback_ = std::move(callback);
    }

    // Get current feature vector (thread-safe snapshot)
    S4FIFOFeatureVector getFeatures() const noexcept;

    // Force a feature update and prediction (for testing)
    void forceFeatureUpdate() noexcept;

    // Check if cache is warmed up
    bool isWarmedUp() const noexcept {
      return isWarmedUp_.load(std::memory_order_acquire);
    }

    // Get the time of last feature update
    uint64_t getLastFeatureUpdateTime() const noexcept {
      return lastFeatureUpdateTime_.load(std::memory_order_acquire);
    }

    class LockedIterator : public Iterator {
     public:
      LockedIterator(const LockedIterator&) = delete;
      LockedIterator& operator=(const LockedIterator&) = delete;
      LockedIterator(LockedIterator&&) noexcept = default;

      void destroy() {
        Iterator::reset();
        if (l_.owns_lock()) {
          l_.unlock();
        }
      }

      void resetToBegin() {
        if (!l_.owns_lock()) {
          l_.lock();
        }
        Iterator::resetToBegin();
      }

     private:
      LockedIterator& operator=(LockedIterator&&) noexcept = default;

      LockedIterator(LockHolder l, const Iterator& iter) noexcept;

      friend Container<T, HookPtr>;

      LockHolder l_;
    };

    Config getConfig() const;
    void setConfig(const Config& newConfig);

    EvictionAgeStat getEvictionAgeStat(uint64_t projectedLength) const noexcept;
    LockedIterator getEvictionIterator() noexcept;

    template <typename F>
    void withEvictionIterator(F&& f);

    template <typename F>
    void withContainerLock(F&& f);

    serialization::MMS4FIFOObject saveState() const noexcept;
    MMContainerStat getStats() const noexcept;

    static LruType getLruType(const T& node) noexcept {
      return isTiny(node) ? LruType::Tiny : LruType::Main;
    }

   private:
    EvictionAgeStat getEvictionAgeStatLocked(
        uint64_t projectedLength) const noexcept;

    static Time getUpdateTime(const T& node) noexcept {
      return (node.*HookPtr).getUpdateTime();
    }

    static void setUpdateTime(T& node, Time time) noexcept {
      (node.*HookPtr).setUpdateTime(time);
    }

    void maybeResizeGhostLocked() noexcept;

    // Promote freq-qualified items from the tiny tail into main until tiny is
    // within its target size or the tiny tail becomes evictable.
    void lazyPromoteTinyTailLocked() noexcept;

    // Batch-reinsert the contiguous freq>0 suffix at the main tail.
    void lazyReinsertMainTailLocked() noexcept;

    static size_t hashNode64(const T& node) noexcept {
      return folly::hasher<folly::StringPiece>()(node.getKey());
    }

    static uint32_t hashNode(const T& node) noexcept {
      return static_cast<uint32_t>(
          folly::hasher<folly::StringPiece>()(node.getKey()));
    }

    void removeLocked(T& node) noexcept;

    // =======================
    // 2-bit frequency counter
    // =======================

    // value = (kMMFlag2 << 1) | kMMFlag1
    static int getFreq(const T& node) noexcept {
      return (node.template isFlagSet<RefFlags::kMMFlag1>() ? 1 : 0) |
            (node.template isFlagSet<RefFlags::kMMFlag2>() ? 2 : 0);
    }

    // set exact value in [0, 3]
    static void setFreq(T& node, int v) noexcept {
      // clamp
      if (v < 0) v = 0;
      if (v > 3) v = 3;

      if (v & 0x1)
        node.template setFlag<RefFlags::kMMFlag1>();
      else
        node.template unSetFlag<RefFlags::kMMFlag1>();

      if (v & 0x2)
        node.template setFlag<RefFlags::kMMFlag2>();
      else
        node.template unSetFlag<RefFlags::kMMFlag2>();
    }

    // saturating increment: 0 → 3
    static void incrementFreq(T& node) noexcept {
      int v = getFreq(node);
      if (v < 3) {
        setFreq(node, v + 1);
      }
    }

    // saturating decrement: 3 → 0
    static void decrementFreq(T& node) noexcept {
      int v = getFreq(node);
      if (v > 0) {
        setFreq(node, v - 1);
      }
    }

    // reset to 0
    static void resetFreq(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag1>();
      node.template unSetFlag<RefFlags::kMMFlag2>();
    }


    static void markTiny(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag0>();
    }

    static void unmarkTiny(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag0>();
    }

    static bool isTiny(const T& node) noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag0>();
    }

    // ========== Feature Collection Private Methods ==========

    void syncRuntimeConfigStateFromConfig() noexcept {
      featureCollectionEnabled_.store(config_.enableFeatureCollection,
                                      std::memory_order_release);
      featureUpdateIntervalSecs_.store(config_.featureUpdateIntervalSecs,
                                       std::memory_order_release);
      periodicUpdatesEnabled_.store(config_.enablePeriodicUpdates,
                                    std::memory_order_release);
      tinySizePercent_.store(config_.tinySizePercent, std::memory_order_release);
      ghostSizePercent_.store(config_.ghostSizePercent,
                              std::memory_order_release);
      moveToMainThreshold_.store(config_.moveToMainThreshold,
                                 std::memory_order_release);
      smallSkipRatio_.store(config_.smallSkipRatio, std::memory_order_release);
      ghostToMainThreshold_.store(config_.ghostToMainThreshold,
                                  std::memory_order_release);
    }

    bool isFeatureCollectionEnabled() const noexcept {
      return featureCollectionEnabled_.load(std::memory_order_acquire);
    }

    uint64_t getFeatureUpdateIntervalSecs() const noexcept {
      return featureUpdateIntervalSecs_.load(std::memory_order_relaxed);
    }

    bool isPeriodicUpdatesEnabled() const noexcept {
      return periodicUpdatesEnabled_.load(std::memory_order_relaxed);
    }

    size_t getTinySizePercent() const noexcept {
      return tinySizePercent_.load(std::memory_order_relaxed);
    }

    size_t getGhostSizePercent() const noexcept {
      return ghostSizePercent_.load(std::memory_order_relaxed);
    }

    int getMoveToMainThreshold() const noexcept {
      return moveToMainThreshold_.load(std::memory_order_relaxed);
    }

    double getSmallSkipRatio() const noexcept {
      return smallSkipRatio_.load(std::memory_order_relaxed);
    }

    int getGhostToMainThreshold() const noexcept {
      return ghostToMainThreshold_.load(std::memory_order_relaxed);
    }

    void initFeatureCollection() {
      syncRuntimeConfigStateFromConfig();
      if (isFeatureCollectionEnabled()) {
        featureCollector_.init(0, 0, 0, 0, config_.featureNumBuckets);
      }
    }

    // Check if it's time for a feature update and perform it if needed
    void maybeUpdateFeatures() noexcept;

    // Apply predicted parameters to config
    void applyPredictedParams(const S4FIFOPredictedParams& params);

    // Default prediction function - returns "no change" for all parameters
    // This means the cache will continue using its current configuration
    static S4FIFOPredictedParams defaultPrediction(
        const S4FIFOFeatureVector& features) {
      (void)features; // unused in placeholder
      // Return params with all sentinel values = no change
      S4FIFOPredictedParams params;
      // All fields already initialized to "no change" sentinel values
      return params;
    }

    mutable folly::cacheline_aligned<Mutex> lruMutex_;

    size_t capacity_{0};
    LruList lru_;
    util::FIFOConcurrentHashSetPair32 ghostQueue_;
    std::atomic<int64_t> sCounter_{0};
    std::atomic<int64_t> mCounter_{0};
    std::atomic<int64_t> gCounter_{0};

    bool shouldLog_{shouldLog};
    Config config_{};
    std::atomic<bool> featureCollectionEnabled_{false};
    std::atomic<uint64_t> featureUpdateIntervalSecs_{0};
    std::atomic<bool> periodicUpdatesEnabled_{false};
    std::atomic<size_t> tinySizePercent_{10};
    std::atomic<size_t> ghostSizePercent_{90};
    std::atomic<int> moveToMainThreshold_{1};
    std::atomic<double> smallSkipRatio_{0.0};
    std::atomic<int> ghostToMainThreshold_{0};

    // ========== Feature Collection State ==========
    S4FIFOFeatureCollector featureCollector_;
    // Use LightGBM predictor by default when feature collection is enabled
    S4FIFOPredictionCallback predictionCallback_{
        facebook::cachelib::lightGBMPredict};

    // Warmup tracking
    std::atomic<bool> isWarmedUp_{false};
    std::atomic<uint64_t> warmupTime_{0};

    // Periodical update tracking
    std::atomic<uint64_t> lastFeatureUpdateTime_{0};
    std::atomic<bool> hasUpdatedOnce_{false};

    // Mutex for feature collection (separate from lru to reduce contention)
    mutable folly::cacheline_aligned<Mutex> featureMutex_;
  };
};

/* Container Interface Implementation */
// This one unused
template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
MMS4FIFO::Container<T, HookPtr>::Container(serialization::MMS4FIFOObject object,
                                           PtrCompressor compressor)
    : lru_(*object.lrus(), std::move(compressor)), config_(*object.config()) {
  initFeatureCollection();
  if (shouldLog_) {
    printf(
        "S4 FIFO Configs are: updateOnWrite=%d, updateOnRead=%d, tinySizePercent=%zu, ghostSizePercent=%zu, skipRatio=%f, moveToMainThreshold=%d, ghostToMainThreshold=%d\n, enableFeatureCollection=%d, featureUpdateIntervalSecs=%lu, enablePeriodicUpdates=%d\n",
        config_.updateOnWrite, config_.updateOnRead,
        config_.tinySizePercent, config_.ghostSizePercent,
        config_.smallSkipRatio, config_.moveToMainThreshold,
        config_.ghostToMainThreshold, config_.enableFeatureCollection,
        config_.featureUpdateIntervalSecs, config_.enablePeriodicUpdates);
  }
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::maybeResizeGhostLocked() noexcept {
  size_t lruSize = lru_.size();
  size_t expectedGhostSize =
      static_cast<size_t>(lruSize * getGhostSizePercent() / 100);

  const bool shouldGrow = lruSize >= 2 * capacity_;
  const bool shouldShrink = lruSize <= capacity_ / 2;
  // If prediction changes ghost size drastically, we also resize
  const bool ghostChangedDrastic = 
      expectedGhostSize >= 1.2 * ghostQueue_.size() ||
      expectedGhostSize <= ghostQueue_.size() * 0.8;

  if (!shouldGrow && !shouldShrink && !ghostChangedDrastic) {
    return;
  }
  ghostQueue_.resize(expectedGhostSize);
  capacity_ = lruSize;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::maybeUpdateFeatures() noexcept {
  if (!isFeatureCollectionEnabled()) {
    return;
  }

  if (!isWarmedUp_.load(std::memory_order_acquire)) {
    return;
  }

  // Check if we should skip updates (one-time mode and already updated)
  if (!isPeriodicUpdatesEnabled() &&
      hasUpdatedOnce_.load(std::memory_order_acquire)) {
    featureCollectionEnabled_.store(false, std::memory_order_release);
    return;
  }

  const auto currTime = static_cast<uint64_t>(util::getCurrentTimeSec());
  // Check if enough time has passed since last update
  uint64_t lastUpdate = lastFeatureUpdateTime_.load(std::memory_order_acquire);
  if (currTime - lastUpdate < getFeatureUpdateIntervalSecs()) {
    return;
  }

  // Time to update! Lock feature mutex
  featureMutex_->lock_combine([this]() {
    // Double-check time under lock
    uint64_t now = static_cast<uint64_t>(util::getCurrentTimeSec());

    uint64_t lastUpdate =
        lastFeatureUpdateTime_.load(std::memory_order_acquire);
    if (now - lastUpdate < getFeatureUpdateIntervalSecs()) {
      return;
    }

    // Get current features
    S4FIFOFeatureVector features;
    featureCollector_.getFeatures(features);
    if (features.totalHits == 0) {
      // Skip predicting for now, no hits collected.
      // Set last update time and return.
      if (shouldLog_) {
        printf(
            "[%lu] S4FIFO Feature Update at time %lu: "
            "No hits collected, skipping prediction.\n",
            config_.tailSize, now);
      }
      lastFeatureUpdateTime_.store(now, std::memory_order_release);
      return;
    }

    // Call prediction function
    S4FIFOPredictedParams predictedParams = predictionCallback_(features);
    if (shouldLog_) {
      printf("[%lu] S4FIFO Feature Vector at time %lu:\n%s\n", config_.tailSize,
            now, featureCollector_.toString().c_str());
      printf(
          "S4FIFO Feature Update at time %lu: "
          "Predicted Params - tinySizePercent=%zu, ghostSizePercent=%zu, "
          "moveToMainThreshold=%d, smallSkipRatio=%.4f, "
          "ghostToMainThreshold=%d\n",
          now,
          predictedParams.tinySizePercent,
          predictedParams.ghostSizePercent,
          predictedParams.moveToMainThreshold,
          predictedParams.smallSkipRatio,
          predictedParams.ghostToMainThreshold);

      // Dump feature vector and prediction into a file, json format
      {
        
        folly::dynamic record = folly::dynamic::object;
        record["time"] = static_cast<int64_t>(now);
        record["tailSize"] = static_cast<int64_t>(config_.tailSize);
        record["featureUpdateInterval"] =
            static_cast<int64_t>(getFeatureUpdateIntervalSecs());
        folly::dynamic featureJson = folly::dynamic::object;
        featureJson["numBuckets"] = features.numBuckets;
        featureJson["logCacheCapacity"] = features.logCacheCapacity;
        featureJson["hitRatioSmall"] = features.hitRatioSmall;
        featureJson["hitRatioMain"] = features.hitRatioMain;
        featureJson["hitRatioGhost"] = features.hitRatioGhost;
        featureJson["uniqueRatio"] = features.uniqueRatio;
        featureJson["oneHitRatio"] = features.oneHitRatio;
        featureJson["totalRequests"] = features.totalRequests;
        featureJson["totalHits"] = features.totalHits;
        featureJson["totalMisses"] = features.totalMisses;
        featureJson["histSmall"] = featureCollector_.histStringify(features.histSmall);
        featureJson["histMain"] = featureCollector_.histStringify(features.histMain);
        featureJson["histGhost"] = featureCollector_.histStringify(features.histGhost);
        record["features"] = featureJson;

        folly::dynamic predictedJson = folly::dynamic::object;
        predictedJson["tinySizePercent"] =
            static_cast<int64_t>(predictedParams.tinySizePercent);
        predictedJson["ghostSizePercent"] =
            static_cast<int64_t>(predictedParams.ghostSizePercent);
        predictedJson["moveToMainThreshold"] =
            static_cast<int64_t>(predictedParams.moveToMainThreshold);
        predictedJson["smallSkipRatio"] = predictedParams.smallSkipRatio;
        predictedJson["ghostToMainThreshold"] =
            static_cast<int64_t>(predictedParams.ghostToMainThreshold);
        record["predictedParams"] = predictedJson;

        std::string recordStr = folly::toJson(record);
        printf("S4FIFO Feature Record: %s\n", recordStr.c_str());
        // Log to file, create a file named "s4fifo_feature_log_<tailSize>.log"
        std::string filename =
            folly::sformat("s4fifo_feature_log_{}.log", config_.tailSize);
        std::ofstream outfile;
        outfile.open(filename, std::ios_base::app);
        outfile << recordStr << std::endl;
        outfile.close();
      }
    }

    // Apply predicted parameters
    applyPredictedParams(predictedParams);

    // Reset feature collector for next period
    featureCollector_.reset();

    // Update timestamps
    lastFeatureUpdateTime_.store(now, std::memory_order_release);
    hasUpdatedOnce_.store(true, std::memory_order_release);
    if (!isPeriodicUpdatesEnabled()) {
      // Disable further updates
      featureCollectionEnabled_.store(false, std::memory_order_release);
    }
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::applyPredictedParams(
    const S4FIFOPredictedParams& params) {
  // Only update parameters that are not set to sentinel values
  // Sentinel values mean "do not change this parameter"
  if (params.tinySizePercent != SIZE_MAX) {
    tinySizePercent_.store(params.tinySizePercent, std::memory_order_release);
  }
  if (params.ghostSizePercent != SIZE_MAX) {
    ghostSizePercent_.store(params.ghostSizePercent, std::memory_order_release);
  }
  if (params.moveToMainThreshold != -1) {
    moveToMainThreshold_.store(params.moveToMainThreshold,
                               std::memory_order_release);
  }
  if (params.smallSkipRatio >= 0.0) {
    smallSkipRatio_.store(params.smallSkipRatio, std::memory_order_release);
  }
  if (params.ghostToMainThreshold != -1) {
    ghostToMainThreshold_.store(params.ghostToMainThreshold,
                                std::memory_order_release);
  }
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
S4FIFOFeatureVector MMS4FIFO::Container<T, HookPtr>::getFeatures()
    const noexcept {
  S4FIFOFeatureVector features;
  featureMutex_->lock_combine(
      [this, &features]() { featureCollector_.getFeatures(features); });
  return features;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::forceFeatureUpdate() noexcept {
  if (!isFeatureCollectionEnabled()) {
    return;
  }

  featureMutex_->lock_combine([this]() {
    S4FIFOFeatureVector features;
    featureCollector_.getFeatures(features);
    S4FIFOPredictedParams predictedParams = predictionCallback_(features);
    applyPredictedParams(predictedParams);
    featureCollector_.reset();
    lastFeatureUpdateTime_.store(
        static_cast<uint64_t>(util::getCurrentTimeSec()),
        std::memory_order_release);
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
bool MMS4FIFO::Container<T, HookPtr>::recordAccess(T& node,
                                                   AccessMode mode) noexcept {
  if ((mode == AccessMode::kWrite && !config_.updateOnWrite) ||
      (mode == AccessMode::kRead && !config_.updateOnRead)) {
    return false;
  }
  // Check for periodical feature update
  maybeUpdateFeatures();

  if (node.isInMMContainer()) {
    // S4FIFO: Check skip ratio for small queue
    const auto smallSkipRatio = getSmallSkipRatio();
    if (UNLIKELY(isTiny(node) && smallSkipRatio > 0)) {
      Time insertTime = getUpdateTime(node);
      Time currentCounter = sCounter_.load(std::memory_order_relaxed);

      int64_t age = currentCounter - static_cast<int64_t>(insertTime);
      int64_t skipThreshold = static_cast<int64_t>(
          smallSkipRatio * lru_.getList(LruType::Tiny).size());

      if (age >= skipThreshold) {
        incrementFreq(node);
      }
    } else {
      incrementFreq(node);
    }

    // Feature collection: record hit
    if (isFeatureCollectionEnabled() &&
        isWarmedUp_.load(std::memory_order_acquire)) {
      featureMutex_->lock_combine([this, &node]() {
        if (isTiny(node)) {
          featureCollector_.totalHitsSmall++;
          featureCollector_.smallTracker.recordHit(
              getUpdateTime(node), 0,
              sCounter_.load(std::memory_order_relaxed));
        } else {
          featureCollector_.totalHitsMain++;
          featureCollector_.mainTracker.recordHit(
              getUpdateTime(node), 0,
              mCounter_.load(std::memory_order_relaxed));
        }
        featureCollector_.totalRequests++;
      });
    }

    return true;
  }
  return false;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
cachelib::EvictionAgeStat MMS4FIFO::Container<T, HookPtr>::getEvictionAgeStat(
    uint64_t projectedLength) const noexcept {
  return lruMutex_->lock_combine([this, projectedLength]() {
    return getEvictionAgeStatLocked(projectedLength);
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
cachelib::EvictionAgeStat
MMS4FIFO::Container<T, HookPtr>::getEvictionAgeStatLocked(
    uint64_t projectedLength) const noexcept {
  if (projectedLength != 0) {
    return EvictionAgeStat{};
  }

  EvictionAgeStat stat;
  const auto curr = static_cast<Time>(util::getCurrentTimeSec());

  auto& list = lru_.getList(LruType::Main);
  auto it = list.rbegin();

  while (it != list.rend() && getFreq(*it) >= 1) {
    ++it;
  }
  stat.warmQueueStat.oldestElementAge =
      it != list.rend() ? curr - getUpdateTime(*it) : 0;

  stat.warmQueueStat.size = list.size();

  return stat;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
bool MMS4FIFO::Container<T, HookPtr>::add(T& node) noexcept {
  const auto nodeHash = hashNode(node);
  const auto ghostContainsWithTS = ghostQueue_.containsWithTS(nodeHash);
  const auto ghostContains = ghostContainsWithTS.first;
  const auto ghostTS = ghostContainsWithTS.second;
  const auto isWarmedUp = isWarmedUp_.load(std::memory_order_acquire);
  const bool featureTrackingEnabled =
      isFeatureCollectionEnabled() && isWarmedUp;

  // Feature collection: record ghost hit
  if (featureTrackingEnabled && ghostContains) {
    featureMutex_->lock_combine([this, ghostTS]() {
      featureCollector_.totalHitsGhost++;
      featureCollector_.ghostTracker.recordHit(
          ghostTS, 0, gCounter_.load(std::memory_order_relaxed));
      gCounter_.fetch_add(1, std::memory_order_relaxed);
      featureCollector_.ghostTracker.recordRemoval();
    });
  }

  bool added = false;
  bool insertedMain = false;
  bool insertedSmall = false;
  bool shouldCountUnique = false;

  {
    LockHolder l(*lruMutex_);

    if (node.isInMMContainer()) {
      return false;
    }

    if (ghostContains) {
      if (getGhostToMainThreshold() <= 0) {
        auto& mainLru = lru_.getList(LruType::Main);
        mainLru.linkAtHead(node);

        unmarkTiny(node);
        setUpdateTime(node, mCounter_.load(std::memory_order_relaxed));
        insertedMain = true;
      } else {
        auto& tinyLru = lru_.getList(LruType::Tiny);
        tinyLru.linkAtHead(node);

        markTiny(node);
        setUpdateTime(node, sCounter_.load(std::memory_order_relaxed));
        insertedSmall = true;
      }
    } else {
      auto& tinyLru = lru_.getList(LruType::Tiny);
      tinyLru.linkAtHead(node);

      markTiny(node);
      setUpdateTime(node, sCounter_.load(std::memory_order_relaxed));
      insertedSmall = true;
      shouldCountUnique = true;
    }

    node.markInMMContainer();
    resetFreq(node);

    added = true;
  }

  // Feature collection: delayed inserts
  if (featureTrackingEnabled && added) {
    featureMutex_->lock_combine([this, insertedMain, insertedSmall,
                                 shouldCountUnique]() {
      if (shouldCountUnique) {
        featureCollector_.totalUnique++;
      }
      featureCollector_.totalMisses++;
      featureCollector_.totalRequests++;

      if (insertedMain) {
        auto mainCounter = mCounter_.load(std::memory_order_relaxed);
        featureCollector_.mainTracker.recordInsert(mainCounter);
        mCounter_.fetch_add(1, std::memory_order_relaxed);
      }

      if (insertedSmall) {
        auto smallCounter = sCounter_.load(std::memory_order_relaxed);
        featureCollector_.smallTracker.recordInsert(smallCounter);
        sCounter_.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  return added;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::lazyReinsertMainTailLocked() noexcept {
  auto& mainList = lru_.getList(LruType::Main);
  auto* tail = mainList.getTail();
  if (tail == nullptr || getFreq(*tail) == 0) {
    return;
  }

  auto* cur = tail;
  auto* first = tail;
  const bool collect =
      isFeatureCollectionEnabled() &&
      isWarmedUp_.load(std::memory_order_acquire);

  if (collect) {
    featureMutex_->lock_combine([this, &mainList, &cur, &first]() {
      while (cur != nullptr && getFreq(*cur) > 0) {
        decrementFreq(*cur);
        const auto mCounterValue = mCounter_.fetch_add(
            1, std::memory_order_relaxed);
        setUpdateTime(*cur, static_cast<Time>(mCounterValue));
        featureCollector_.mainTracker.recordInsert(mCounterValue);
        first = cur;
        cur = mainList.getPrev(*cur);
      }
    });
  } else {
    while (cur != nullptr && getFreq(*cur) > 0) {
      decrementFreq(*cur);
      first = cur;
      cur = mainList.getPrev(*cur);
    }
  }

  mainList.moveSuffixToHead(*first);
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::lazyPromoteTinyTailLocked() noexcept {
  auto& tinyLru = lru_.getList(LruType::Tiny);
  auto& mainLru = lru_.getList(LruType::Main);

  auto totalSize = tinyLru.size() + mainLru.size();
  auto targetTinySize =
      static_cast<size_t>(getTinySizePercent() * totalSize / 100);

  const bool featureTrackingEnabled =
      isFeatureCollectionEnabled() &&
      isWarmedUp_.load(std::memory_order_acquire);

  // Only touch tiny when it exceeds its target size. At steady state this
  // stays constant time.
  while (tinyLru.size() > targetTinySize) {
    T* nodePtr = tinyLru.getTail();
    if (nodePtr == nullptr) {
      break;
    }

    T& node = *nodePtr;
    if (getFreq(node) < getMoveToMainThreshold()) {
      break;
    }

    tinyLru.remove(node);
    mainLru.linkAtHead(node);
    unmarkTiny(node);
    resetFreq(node);
    if (featureTrackingEnabled) {
      auto mCounterValue = mCounter_.load(std::memory_order_relaxed);
      setUpdateTime(node, mCounterValue);
      featureMutex_->lock_combine([this, mCounterValue]() {
        featureCollector_.mainTracker.recordInsert(mCounterValue);
        mCounter_.fetch_add(1, std::memory_order_relaxed);
      });
    }
  }
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
typename MMS4FIFO::Container<T, HookPtr>::LockedIterator
MMS4FIFO::Container<T, HookPtr>::getEvictionIterator() noexcept {
  LockHolder l(*lruMutex_);
  maybeResizeGhostLocked();
  lazyPromoteTinyTailLocked();

  const auto totalSize = lru_.size();
  const auto targetTinySize =
      totalSize == 0 ? 0 : totalSize * getTinySizePercent() / 100;
  if (lru_.getList(LruType::Tiny).size() > targetTinySize ||
      lru_.getList(LruType::Main).size() == 0) {
    return LockedIterator{
        std::move(l), lru_.getList(LruType::Tiny).rbegin()};
  }

  lazyReinsertMainTailLocked();
  return LockedIterator{std::move(l), lru_.getList(LruType::Main).rbegin()};
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
template <typename F>
void MMS4FIFO::Container<T, HookPtr>::withEvictionIterator(F&& fun) {
  fun(getEvictionIterator());
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
template <typename F>
void MMS4FIFO::Container<T, HookPtr>::withContainerLock(F&& fun) {
  lruMutex_->lock_combine([&fun]() { fun(); });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::removeLocked(T& node) noexcept {
  if (isTiny(node)) {
    lru_.getList(LruType::Tiny).remove(node);
    unmarkTiny(node);

    // Feature collection: record one-hit wonder
    if (isFeatureCollectionEnabled() &&
        isWarmedUp_.load(std::memory_order_acquire) &&
        getFreq(node) < getMoveToMainThreshold()) {
      featureMutex_->lock_combine(
          [this]() { featureCollector_.oneHitCount++; });
    }
  } else {
    lru_.getList(LruType::Main).remove(node);
  }

  resetFreq(node);
  node.unmarkInMMContainer();
  return;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
bool MMS4FIFO::Container<T, HookPtr>::remove(T& node) noexcept {
  bool isTiny_ = isTiny(node);
  auto result = lruMutex_->lock_combine([this, &node]() {
    if (!node.isInMMContainer()) {
      return false;
    }
    removeLocked(node);
    return true;
  });

  if (result && isTiny_) {
    auto gCounterValue = gCounter_.load(std::memory_order_relaxed);
    ghostQueue_.insert(hashNode(node), gCounterValue);
    if (isFeatureCollectionEnabled() &&
        isWarmedUp_.load(std::memory_order_acquire)) {
      featureMutex_->lock_combine([this, gCounterValue]() {
        featureCollector_.ghostTracker.recordInsert(gCounterValue);
        gCounter_.fetch_add(1, std::memory_order_relaxed);
      });
    }
  }
  return result;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::remove(LockedIterator& it) noexcept {
  T& node = *it;
  XDCHECK(node.isInMMContainer());
  ++it;

  bool evictedFromTiny = false;
  const int freqBeforeReset = getFreq(node);
  bool shouldWarmUp = false;
  size_t warmupTotalSize = 0;
  size_t warmupTinySizeReal = 0;
  size_t warmupMainSizeReal = 0;
  size_t warmupGhostSize = 0;
  uint64_t warmupLastFeatureUpdate = 0;
  uint64_t warmupTimeSec = 0;
  if (isTiny(node)) {
    lru_.getList(LruType::Tiny).remove(node);
    unmarkTiny(node);
    evictedFromTiny = true;

    // Record one hit wonder, delay to after unlock
  } else {
    lru_.getList(LruType::Main).remove(node);
  }

  resetFreq(node);
  node.unmarkInMMContainer();

  if (evictedFromTiny) {
    if (isFeatureCollectionEnabled() &&
        !isWarmedUp_.load(std::memory_order_acquire)) {
      warmupTotalSize = lru_.size();
      warmupTinySizeReal = lru_.getList(LruType::Tiny).size();
      warmupMainSizeReal = lru_.getList(LruType::Main).size();
      const size_t tinySize =
          static_cast<size_t>(getTinySizePercent() * warmupTotalSize / 100);
      const bool isTinyWithRealWithinTarget =
          tinySize * 1.01 >= warmupTinySizeReal;
      if (warmupTotalSize >= kS4FIFOMinTrackedLruSize &&
          isTinyWithRealWithinTarget) {
        warmupGhostSize =
            warmupTotalSize * getGhostSizePercent() / 100;
        warmupLastFeatureUpdate =
            lastFeatureUpdateTime_.load(std::memory_order_acquire);
        warmupTimeSec = static_cast<uint64_t>(util::getCurrentTimeSec());
        shouldWarmUp = true;
      }
    }

    if (it.l_.owns_lock()) {
      it.l_.unlock();
    }
    // Feature collection: record one-hit wonder
    if (isFeatureCollectionEnabled() &&
        isWarmedUp_.load(std::memory_order_acquire) &&
        freqBeforeReset < getMoveToMainThreshold()) {
      featureMutex_->lock_combine(
          [this]() { featureCollector_.oneHitCount++; });
    }

    // Insert to ghost, with the ghostcounter value
    auto gCounterValue = gCounter_.load(std::memory_order_relaxed);
    ghostQueue_.insert(hashNode(node), gCounterValue);

    if (isFeatureCollectionEnabled() &&
        isWarmedUp_.load(std::memory_order_acquire)) {
      featureMutex_->lock_combine([this, gCounterValue]() {
        featureCollector_.ghostTracker.recordInsert(gCounterValue);
        gCounter_.fetch_add(1, std::memory_order_relaxed);
      });
    }

    if (shouldWarmUp) {
      featureMutex_->lock_combine([this, warmupTotalSize, warmupTinySizeReal,
                                   warmupMainSizeReal, warmupGhostSize,
                                   warmupLastFeatureUpdate, warmupTimeSec]() {
        if (isWarmedUp_.load(std::memory_order_acquire)) {
          return;
        }
        if (shouldLog_) {
          printf(
              "[%lu] S4FIFO Cache Warmed Up at time %lu with size %zu, tailsize %lu, last feature "
              "update at %lu\n",
              config_.tailSize,
              warmupTimeSec,
              warmupTotalSize,
              config_.tailSize,
              warmupLastFeatureUpdate);
        }
        featureCollector_.init(warmupTotalSize,
                               warmupTinySizeReal,
                               warmupMainSizeReal,
                               warmupGhostSize,
                               config_.featureNumBuckets);
        lastFeatureUpdateTime_.store(warmupTimeSec, std::memory_order_release);
        warmupTime_.store(warmupTimeSec, std::memory_order_release);
        featureCollector_.setWarmedUp();
        isWarmedUp_.store(true, std::memory_order_release);
      });
    }
  }
  return;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
bool MMS4FIFO::Container<T, HookPtr>::replace(T& oldNode, T& newNode) noexcept {
  return lruMutex_->lock_combine([this, &oldNode, &newNode]() {
    if (!oldNode.isInMMContainer() || newNode.isInMMContainer()) {
      return false;
    }
    const auto updateTime = getUpdateTime(oldNode);

    if (isTiny(oldNode)) {
      lru_.getList(LruType::Tiny).replace(oldNode, newNode);
      unmarkTiny(oldNode);
      markTiny(newNode);
    } else {
      lru_.getList(LruType::Main).replace(oldNode, newNode);
    }

    oldNode.unmarkInMMContainer();
    newNode.markInMMContainer();
    setUpdateTime(newNode, updateTime);
    setFreq(newNode, getFreq(oldNode));
    return true;
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
typename MMS4FIFO::Config MMS4FIFO::Container<T, HookPtr>::getConfig() const {
  return lruMutex_->lock_combine([this]() {
    auto config = config_;
    config.enableFeatureCollection = isFeatureCollectionEnabled();
    config.featureUpdateIntervalSecs = getFeatureUpdateIntervalSecs();
    config.enablePeriodicUpdates = isPeriodicUpdatesEnabled();
    config.tinySizePercent = getTinySizePercent();
    config.ghostSizePercent = getGhostSizePercent();
    config.moveToMainThreshold = getMoveToMainThreshold();
    config.smallSkipRatio = getSmallSkipRatio();
    config.ghostToMainThreshold = getGhostToMainThreshold();
    return config;
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::setConfig(const Config& c) {
  lruMutex_->lock_combine([this, c]() {
    config_ = c;
    syncRuntimeConfigStateFromConfig();
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
serialization::MMS4FIFOObject MMS4FIFO::Container<T, HookPtr>::saveState()
    const noexcept {
  serialization::MMS4FIFOConfig configObject;
  *configObject.updateOnWrite() = config_.updateOnWrite;
  *configObject.updateOnRead() = config_.updateOnRead;
  *configObject.ghostSizePercent() = getGhostSizePercent();
  *configObject.tinySizePercent() = getTinySizePercent();
  *configObject.moveToMainThreshold() = getMoveToMainThreshold();
  *configObject.smallSkipRatio() = getSmallSkipRatio();
  *configObject.ghostToMainThreshold() = getGhostToMainThreshold();

  serialization::MMS4FIFOObject object;
  *object.config() = configObject;
  *object.lrus() = lru_.saveState();
  return object;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
MMContainerStat MMS4FIFO::Container<T, HookPtr>::getStats() const noexcept {
  return {lru_.size(), 0, 0, 0, 0, 0, 0};
}

// Locked Iterator Context Implementation
template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
MMS4FIFO::Container<T, HookPtr>::LockedIterator::LockedIterator(
    LockHolder l, const Iterator& iter) noexcept
    : Iterator(iter), l_(std::move(l)) {}
} // namespace facebook::cachelib
