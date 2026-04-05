// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/// Minimal fork-join thread pool for parallel_for workloads.
///
/// Workers are created once and reused across ForEach calls.
/// The calling thread participates as thread 0.
/// Not intended for general task queuing.
class ThreadPool
{
public:
    /// Create pool with num_workers background threads (0 = auto-detect).
    explicit ThreadPool(size_t num_workers = 0)
    {
        if (num_workers == 0)
        {
            const unsigned hw = std::thread::hardware_concurrency();
            num_workers = (hw > 1) ? hw - 1 : 0;
        }

        workers_.reserve(num_workers);
        for (size_t i = 0; i < num_workers; ++i)
            workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }

    ~ThreadPool()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Total thread count (workers + caller).
    [[nodiscard]] size_t NumThreads() const { return workers_.size() + 1; }

    /// Execute func(thread_id, range_begin, range_end) for chunks of [0, count).
    /// Caller participates as thread 0. Blocks until all work completes.
    template <typename F>
    void ForEach(size_t count, F&& func)
    {
        if (count == 0) return;

        if (workers_.empty())
        {
            func(size_t{0}, size_t{0}, count);
            return;
        }

        const size_t nt = NumThreads();
        const size_t chunk = (count + nt - 1) / nt;

        // Publish work for background workers
        {
            std::lock_guard lock(mutex_);
            for_func_ = [&func, chunk, count](size_t tid) {
                const size_t b = tid * chunk;
                if (b >= count) return;
                func(tid, b, std::min(b + chunk, count));
            };
            remaining_.store(static_cast<int>(workers_.size()));
            ++generation_;
        }
        cv_work_.notify_all();

        // Caller handles chunk 0
        func(size_t{0}, size_t{0}, std::min(chunk, count));

        // Wait for all workers to finish
        std::unique_lock lock(mutex_);
        cv_done_.wait(lock, [this] { return remaining_.load() == 0; });
        for_func_ = nullptr; // release captured references
    }

private:
    void WorkerLoop(size_t tid)
    {
        size_t local_gen = 0;
        while (true)
        {
            {
                std::unique_lock lock(mutex_);
                cv_work_.wait(lock, [&] { return stop_ || generation_ > local_gen; });
                if (stop_) return;
                local_gen = generation_;
            }

            for_func_(tid);

            if (remaining_.fetch_sub(1) == 1)
                cv_done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    std::function<void(size_t)> for_func_;
    std::atomic<int> remaining_{0};
    size_t generation_ = 0;
    bool stop_ = false;
};
