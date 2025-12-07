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
#include <functional>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <folly/Format.h>
#include <folly/Math.h>
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

  void init(int64_t expectedMaxPos, int32_t buckets, bool trackRemoval) {
    memset(this, 0, sizeof(S4FIFOHitPosTracker));
    if (buckets <= 0)
      buckets = kS4FIFODefaultBuckets;
    if (buckets > kS4FIFOMaxBuckets)
      buckets = kS4FIFOMaxBuckets;
    numBuckets = buckets;
    bucketSize = (expectedMaxPos + buckets - 1) / buckets;
    if (bucketSize < 1)
      bucketSize = 1;
    trackMiddleRemoval = trackRemoval;
  }

  int64_t recordInsert(int64_t insertCounter) {
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

  void recordRemoval() {
    if (!trackMiddleRemoval)
      return;
    removalCounters[currentBucket]++;
    totalRemovals++;
  }

  int64_t estimateHoles(int64_t insertBucket) const {
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

  void recordHit(int64_t insertTime,
                 int64_t insertBucket,
                 int64_t currentCounter) {
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

    Config() = default;
    Config(const Config& rhs) = default;
    Config(Config&& rhs) = default;

    Config& operator=(const Config& rhs) = default;
    Config& operator=(Config&& rhs) = default;

    template <typename... Args>
    void addExtraConfig(Args...) {}

    // whether the cache needs to be updated on writes for recordAccess.
    bool updateOnWrite{false};

    // whether the cache needs to be updated on reads for recordAccess.
    bool updateOnRead{true};

    // The size of tiny cache, as a percentage of the total size.
    size_t tinySizePercent{10};

    // The size of ghost queue, as a percentage of total size
    size_t ghostSizePercent{90};

    // S4FIFO-specific: frequency threshold for small->main promotion
    int moveToMainThreshold{2};

    // S4FIFO-specific: ratio to skip frequency increment in small queue
    double smallSkipRatio{0.0};

    // S4FIFO-specific: frequency threshold for ghost->main promotion
    int ghostToMainThreshold{0};

    // ========== Feature Collection & Prediction Settings ==========

    // Enable feature collection (default: false)
    bool enableFeatureCollection{false};

    // Time interval in seconds for periodical feature updates (default: 1440s =
    // 24min) After warmup, features are collected and prediction is called
    // every this interval
    uint64_t featureUpdateIntervalSecs{1440};

    // If true, continuously update parameters periodically
    // If false, only update once after first interval
    bool enablePeriodicUpdates{true};

    // Number of histogram buckets for feature collection
    int32_t featureNumBuckets{kS4FIFODefaultBuckets};
  };

  // The container object which can be used to keep track of objects of type T
  template <typename T, Hook<T> T::* HookPtr>
  struct Container {
   private:
    using LruList = MultiDList<T, HookPtr>;
    using Mutex = folly::DistributedMutex;
    using LockHolder = std::unique_lock<Mutex>;
    using PtrCompressor = typename T::PtrCompressor;
    using Time = typename Hook<T>::Time;
    using CompressedPtrType = typename T::CompressedPtrType;
    using RefFlags = typename T::Flags;

   public:
    Container() {};
    Container(Config c, PtrCompressor compressor)
        : lru_(LruType::NumTypes, std::move(compressor)),
          config_(std::move(c)) {
      initFeatureCollection();
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

    class LockedIterator {
     public:
      using ListIterator = typename LruList::DListIterator;
      LockedIterator(const LockedIterator&) = delete;
      LockedIterator& operator=(const LockedIterator&) = delete;
      LockedIterator(LockedIterator&&) noexcept = default;

      LockedIterator& operator++() noexcept {
        ListIterator& it = getIter();
        if (!it) {
          return *this;
        }
        ++it;
        skipPromotable(it);
        return *this;
      }

      ListIterator& skipPromotable(ListIterator& it) noexcept {
        while (it) {
          T& node = *it;
          if (!shouldPromote(node)) {
            break;
          }
          ++it;
        }
        return it;
      }

      LockedIterator& operator--() {
        throw std::invalid_argument(
            "Decrementing eviction iterator is not supported");
      }

      T* operator->() const noexcept { return getIter().operator->(); }
      T& operator*() const noexcept { return getIter().operator*(); }

      bool operator==(const LockedIterator& other) const noexcept {
        return &c_ == &other.c_ && tIter_ == other.tIter_ &&
               mIter_ == other.mIter_;
      }

      bool operator!=(const LockedIterator& other) const noexcept {
        return !(*this == other);
      }

      explicit operator bool() const noexcept { return tIter_ || mIter_; }

      T* get() const noexcept { return getIter().get(); }

      void reset() noexcept {
        tIter_.reset();
        mIter_.reset();
      }

      void destroy() {
        reset();
        if (l_.owns_lock()) {
          l_.unlock();
        }
      }

      void resetToBegin() {
        if (!l_.owns_lock()) {
          l_.lock();
        }
        tIter_.resetToBegin();
        mIter_.resetToBegin();
      }

     private:
      LockedIterator& operator=(LockedIterator&&) noexcept = default;

      explicit LockedIterator(LockHolder l,
                              const Container<T, HookPtr>& c) noexcept;

      const ListIterator& getIter() const noexcept {
        auto shouldEvictTiny = evictTiny();
        return shouldEvictTiny ? tIter_ : mIter_;
      }

      ListIterator& getIter() noexcept {
        return const_cast<ListIterator&>(
            static_cast<const LockedIterator*>(this)->getIter());
      }

      bool shouldPromote(const T& node) const noexcept {
        if (Container<T, HookPtr>::isTiny(node)) {
          return Container<T, HookPtr>::getFreq(node) >=
                 c_.config_.moveToMainThreshold;
        } else {
          return Container<T, HookPtr>::getFreq(node) >= 1;
        }
      }

      bool evictTiny() const noexcept {
        if (!mIter_) {
          return true;
        }
        if (!tIter_) {
          return false;
        }
        if (evictTinyCache_ != -1) {
          return evictTinyCache_ == 1;
        }

        int smallSize = static_cast<int>(c_.lru_.getList(LruType::Tiny).size());
        const size_t targetTiny = static_cast<size_t>(
            c_.config_.tinySizePercent * c_.lru_.size() / 100);

        this->evictTinyCache_ = smallSize >= targetTiny;
        return evictTinyCache_;
      }

      mutable int evictTinyCache_{-1};

      friend Container<T, HookPtr>;

      const Container<T, HookPtr>& c_;
      ListIterator tIter_;
      ListIterator mIter_;
      LockHolder l_;
    };

    Config getConfig() const;
    void setConfig(const Config& newConfig);

    EvictionAgeStat getEvictionAgeStat(uint64_t projectedLength) const noexcept;
    LockedIterator getEvictionIterator() noexcept;

    void rebalanceForEviction();

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

    static size_t hashNode64(const T& node) noexcept {
      return folly::hasher<folly::StringPiece>()(node.getKey());
    }

    static uint32_t hashNode(const T& node) noexcept {
      return static_cast<uint32_t>(
          folly::hasher<folly::StringPiece>()(node.getKey()));
    }

    void removeLocked(T& node) noexcept;

    static bool isTiny(const T& node) noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag0>();
    }

    static int getFreq(const T& node) noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag1>() ? 1 : 0;
    }

    static void incrementFreq(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag1>();
    }

    static void resetFreq(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag1>();
    }

    static bool isAccessed(const T& node) noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag1>();
    }

    static void markTiny(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag0>();
    }
    static void unmarkTiny(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag0>();
    }

    static void markAccessed(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag1>();
    }
    static void unmarkAccessed(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag1>();
    }

    // ========== Feature Collection Private Methods ==========

    void initFeatureCollection() {
      if (config_.enableFeatureCollection) {
        // Will be properly initialized when we know sizes
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
    facebook::cachelib::util::FIFOConcurrentHashSet32 ghostQueue_;
    std::atomic<int64_t> sCounter_{0};
    Config config_{};

    // ========== Feature Collection State ==========
    S4FIFOFeatureCollector featureCollector_;
    S4FIFOPredictionCallback predictionCallback_{defaultPrediction};

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
template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
MMS4FIFO::Container<T, HookPtr>::Container(serialization::MMS4FIFOObject object,
                                           PtrCompressor compressor)
    : lru_(*object.lrus(), std::move(compressor)), config_(*object.config()) {
  initFeatureCollection();
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::maybeResizeGhostLocked() noexcept {
  size_t lruSize = lru_.size();

  const bool shouldGrow = lruSize >= 2 * capacity_;
  const bool shouldShrink = lruSize <= capacity_ / 2;

  if (!shouldGrow && !shouldShrink) {
    return;
  }

  size_t expectedGhostSize =
      static_cast<size_t>(lruSize * config_.ghostSizePercent / 100);
  ghostQueue_.resize(expectedGhostSize);
  capacity_ = lruSize;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::maybeUpdateFeatures() noexcept {
  if (!config_.enableFeatureCollection) {
    return;
  }

  const auto currTime = static_cast<uint64_t>(util::getCurrentTimeSec());

  // Check warmup: cache is warmed up when it's full
  if (!isWarmedUp_.load(std::memory_order_acquire)) {
    // Check if we're warmed up (simplified check - could also use occupied
    // bytes)
    if (lru_.size() > 0) {
      isWarmedUp_.store(true, std::memory_order_release);
      warmupTime_.store(currTime, std::memory_order_release);
      lastFeatureUpdateTime_.store(currTime, std::memory_order_release);

      // Initialize feature collector with actual sizes
      size_t tinySize = lru_.getList(LruType::Tiny).size();
      size_t mainSize = lru_.getList(LruType::Main).size();
      size_t totalSize = tinySize + mainSize;
      featureCollector_.init(totalSize,
                             tinySize,
                             mainSize,
                             totalSize * config_.ghostSizePercent / 100,
                             config_.featureNumBuckets);
    }
    return;
  }

  // Check if we should skip updates (one-time mode and already updated)
  if (!config_.enablePeriodicUpdates &&
      hasUpdatedOnce_.load(std::memory_order_acquire)) {
    return;
  }

  // Check if enough time has passed since last update
  uint64_t lastUpdate = lastFeatureUpdateTime_.load(std::memory_order_acquire);
  if (currTime - lastUpdate < config_.featureUpdateIntervalSecs) {
    return;
  }

  // Time to update! Lock feature mutex
  featureMutex_->lock_combine([this, currTime]() {
    // Double-check time under lock
    uint64_t lastUpdate =
        lastFeatureUpdateTime_.load(std::memory_order_relaxed);
    uint64_t now = static_cast<uint64_t>(util::getCurrentTimeSec());

    if (now - lastUpdate < config_.featureUpdateIntervalSecs) {
      return;
    }

    // Get current features
    S4FIFOFeatureVector features;
    featureCollector_.getFeatures(features);

    // Call prediction function
    S4FIFOPredictedParams predictedParams = predictionCallback_(features);

    // Apply predicted parameters
    applyPredictedParams(predictedParams);

    // Reset feature collector for next period
    featureCollector_.reset();

    // Update timestamps
    lastFeatureUpdateTime_.store(now, std::memory_order_release);
    hasUpdatedOnce_.store(true, std::memory_order_release);
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::applyPredictedParams(
    const S4FIFOPredictedParams& params) {
  // Only update parameters that are not set to sentinel values
  // Sentinel values mean "do not change this parameter"
  if (params.tinySizePercent != SIZE_MAX) {
    config_.tinySizePercent = params.tinySizePercent;
  }
  if (params.ghostSizePercent != SIZE_MAX) {
    config_.ghostSizePercent = params.ghostSizePercent;
  }
  if (params.moveToMainThreshold != -1) {
    config_.moveToMainThreshold = params.moveToMainThreshold;
  }
  if (params.smallSkipRatio >= 0.0) {
    config_.smallSkipRatio = params.smallSkipRatio;
  }
  if (params.ghostToMainThreshold != -1) {
    config_.ghostToMainThreshold = params.ghostToMainThreshold;
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
  if (!config_.enableFeatureCollection) {
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
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());

  // Check for periodical feature update
  maybeUpdateFeatures();

  if (node.isInMMContainer()) {
    // S4FIFO: Check skip ratio for small queue
    if (isTiny(node) && config_.smallSkipRatio > 0) {
      int64_t currentCounter = sCounter_.load(std::memory_order_relaxed);
      Time insertTime = getUpdateTime(node);
      int64_t age = currentCounter - static_cast<int64_t>(insertTime);
      int64_t skipThreshold = static_cast<int64_t>(
          config_.smallSkipRatio * lru_.getList(LruType::Tiny).size());

      if (age >= skipThreshold) {
        incrementFreq(node);
      }
    } else {
      incrementFreq(node);
    }
    setUpdateTime(node, currTime);

    // Feature collection: record hit
    if (config_.enableFeatureCollection &&
        isWarmedUp_.load(std::memory_order_acquire)) {
      featureMutex_->lock_combine([this, &node]() {
        if (isTiny(node)) {
          featureCollector_.totalHitsSmall++;
        } else {
          featureCollector_.totalHitsMain++;
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
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());

  const auto nodeHash = hashNode(node);
  const auto ghostContains = ghostQueue_.contains(nodeHash);

  // Feature collection: record ghost hit
  if (config_.enableFeatureCollection && ghostContains &&
      isWarmedUp_.load(std::memory_order_acquire)) {
    featureMutex_->lock_combine(
        [this]() { featureCollector_.totalHitsGhost++; });
  }

  return lruMutex_->lock_combine([this, &node, currTime, ghostContains]() {
    if (node.isInMMContainer()) {
      return false;
    }

    if (ghostContains) {
      if (config_.ghostToMainThreshold <= 0) {
        auto& mainLru = lru_.getList(LruType::Main);
        mainLru.linkAtHead(node);
      } else {
        auto& tinyLru = lru_.getList(LruType::Tiny);
        tinyLru.linkAtHead(node);
        markTiny(node);
        sCounter_.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      auto& tinyLru = lru_.getList(LruType::Tiny);
      tinyLru.linkAtHead(node);
      markTiny(node);
      sCounter_.fetch_add(1, std::memory_order_relaxed);

      // Feature collection: record unique object
      if (config_.enableFeatureCollection &&
          isWarmedUp_.load(std::memory_order_acquire)) {
        featureCollector_.totalUnique++;
      }
    }

    node.markInMMContainer();
    setUpdateTime(node, currTime);
    resetFreq(node);

    // Feature collection: record miss (new insertion = cache miss)
    if (config_.enableFeatureCollection &&
        isWarmedUp_.load(std::memory_order_acquire)) {
      featureCollector_.totalMisses++;
      featureCollector_.totalRequests++;
    }

    return true;
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::rebalanceForEviction() {
  auto& tinyLru = lru_.getList(LruType::Tiny);
  auto& mainLru = lru_.getList(LruType::Main);

  auto totalSize = tinyLru.size() + mainLru.size();
  auto expectedTinySize =
      static_cast<size_t>(config_.tinySizePercent * totalSize / 100);

  while (true) {
    bool tryTiny = tinyLru.size() >= expectedTinySize;

    if (tryTiny) {
      T* nodePtr = tinyLru.getTail();
      if (!nodePtr) {
        break;
      }

      T& node = *nodePtr;
      if (getFreq(node) >= config_.moveToMainThreshold) {
        tinyLru.remove(node);
        mainLru.linkAtHead(node);
        unmarkTiny(node);
        resetFreq(node);
        continue;
      } else {
        break;
      }
    } else {
      T* nodePtr = mainLru.getTail();
      if (!nodePtr) {
        break;
      }

      T& node = *nodePtr;
      if (getFreq(node) >= 1) {
        mainLru.moveToHead(node);
        resetFreq(node);
        continue;
      } else {
        break;
      }
    }
  }
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
typename MMS4FIFO::Container<T, HookPtr>::LockedIterator
MMS4FIFO::Container<T, HookPtr>::getEvictionIterator() noexcept {
  LockHolder l(*lruMutex_);
  maybeResizeGhostLocked();
  rebalanceForEviction();
  return LockedIterator{std::move(l), *this};
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
    if (config_.enableFeatureCollection &&
        isWarmedUp_.load(std::memory_order_acquire) &&
        getFreq(node) < config_.moveToMainThreshold) {
      featureCollector_.oneHitCount++;
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
    ghostQueue_.insert(hashNode(node));
  }
  return result;
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::remove(LockedIterator& it) noexcept {
  T& node = *it;
  XDCHECK(node.isInMMContainer());
  ++it;

  bool evictedFromTiny = false;
  if (isTiny(node)) {
    lru_.getList(LruType::Tiny).remove(node);
    unmarkTiny(node);
    evictedFromTiny = true;

    // Feature collection: record one-hit wonder
    if (config_.enableFeatureCollection &&
        isWarmedUp_.load(std::memory_order_acquire) &&
        getFreq(node) < config_.moveToMainThreshold) {
      featureCollector_.oneHitCount++;
    }
  } else {
    lru_.getList(LruType::Main).remove(node);
  }

  resetFreq(node);
  node.unmarkInMMContainer();

  if (evictedFromTiny) {
    if (it.l_.owns_lock()) {
      it.l_.unlock();
    }
    ghostQueue_.insert(hashNode(node));
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
    if (isAccessed(oldNode)) {
      markAccessed(newNode);
    } else {
      unmarkAccessed(newNode);
    }
    return true;
  });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
typename MMS4FIFO::Config MMS4FIFO::Container<T, HookPtr>::getConfig() const {
  return lruMutex_->lock_combine([this]() { return config_; });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
void MMS4FIFO::Container<T, HookPtr>::setConfig(const Config& c) {
  lruMutex_->lock_combine([this, c]() { config_ = c; });
}

template <typename T, MMS4FIFO::Hook<T> T::* HookPtr>
serialization::MMS4FIFOObject MMS4FIFO::Container<T, HookPtr>::saveState()
    const noexcept {
  serialization::MMS4FIFOConfig configObject;
  *configObject.updateOnWrite() = config_.updateOnWrite;
  *configObject.updateOnRead() = config_.updateOnRead;
  *configObject.ghostSizePercent() = config_.ghostSizePercent;
  *configObject.tinySizePercent() = config_.tinySizePercent;
  *configObject.moveToMainThreshold() = config_.moveToMainThreshold;
  *configObject.smallSkipRatio() = config_.smallSkipRatio;
  *configObject.ghostToMainThreshold() = config_.ghostToMainThreshold;

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
    LockHolder l, const Container<T, HookPtr>& c) noexcept
    : c_(c),
      tIter_(c.lru_.getList(LruType::Tiny).rbegin()),
      mIter_(c.lru_.getList(LruType::Main).rbegin()),
      l_(std::move(l)) {}
} // namespace facebook::cachelib
