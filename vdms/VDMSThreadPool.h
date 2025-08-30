#ifndef VDMS_THREAD_POOL_H
#define VDMS_THREAD_POOL_H

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class VDMSThreadPool {
 public:
  static VDMSThreadPool& instance();

  // Enqueues a task to be executed by a worker thread.
  template <class F>
  void enqueue(F&& f);

  // Let's prevents copy and move operations so we don't end up with multiple
  // thread pools
  VDMSThreadPool(const VDMSThreadPool&) = delete;
  VDMSThreadPool& operator=(const VDMSThreadPool&) = delete;
  VDMSThreadPool(VDMSThreadPool&&) = delete;
  VDMSThreadPool& operator=(VDMSThreadPool&&) = delete;

 private:
  explicit VDMSThreadPool(size_t threads);

  ~VDMSThreadPool();

  // The collection of worker threads managed by the pool.
  std::vector<std::thread> workers;

  // queue for tasks
  std::queue<std::function<void()>> tasks;

  // Synchronization variables
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// Constructor to create a specified number of worker threads.
// creating more than the number of SMT cores will add overhead un-necassarily
// if one of the threads crashes for any reason it will log the crashing reason
// and then a new thread is created in its place
inline VDMSThreadPool::VDMSThreadPool(size_t threads) : stop(false) {
  if (threads == 0) {
    throw std::invalid_argument("Thread pool must have at least one thread.");
  }

  auto worker_task = [this] {
    while (!stop) {
      try {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });

            if (this->stop && this->tasks.empty()) {
              return;
            }

            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      } catch (const std::exception& e) {
        std::cerr << "Worker thread caught std::exception and will restart: "
                  << e.what() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } catch (...) {
        std::cerr << "Worker thread caught unknown exception and will restart."
                  << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  };

  for (size_t i = 0; i < threads; ++i) {
    workers.emplace_back(worker_task);
  }
}

// The destructor signals all threads to stop and waits for them to finish.
inline VDMSThreadPool::~VDMSThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();  // Wake up all waiting threads.
  for (std::thread& worker : workers) {
    if (worker.joinable()) {
      worker.join();  // Wait for each thread to complete.
    }
  }
}

inline VDMSThreadPool& VDMSThreadPool::instance() {
  static VDMSThreadPool pool(4);
  // std::thread::hardware_concurrency());  // This creates number of threads =
  //                                        // number of SMT cores
  return pool;
}

// Enqueues a task to the pool.
template <class F>
inline void VDMSThreadPool::enqueue(F&& f) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    // Use emplace to add the task directly in the queue.
    tasks.emplace(std::forward<F>(f));
  }  // Lock is released.

  // Notify one waiting thread that a new task is ready.
  condition.notify_one();
}

#endif  // VDMS_THREAD_POOL_H