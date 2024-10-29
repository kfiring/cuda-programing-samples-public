#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <assert.h>


/**
 * 在CPU上模拟CUDA kernel运行方式的模拟器
 */
class CPUKernelSimulator {
public:
    /**
     * 线程上下文结构，包含总线程数，当前线程索引信息，模仿blockDim，threadIdx等
     */
    struct Context
    {
        uint total_thread_count, thread_idx;
        Context(const uint& thread_count, const uint& idx) 
            : total_thread_count(thread_count), thread_idx(idx) {}
    };
    
    explicit inline CPUKernelSimulator(uint thread_count=std::thread::hardware_concurrency()) 
        : _launched(false), _stop(false), _finish_flag(thread_count, false) {
        assert(thread_count > 0);
        printf("cpu kernel simulator with %d threads\n", thread_count);

        // 事先把线程都创建好
        for (uint i = 0; i < thread_count; i++) {
            _threads.emplace_back(std::thread([this, thread_count, i](){
                // 线程函数
                // 构建线程上下文变量
                thread_local Context ctxt(thread_count, i);

                // 线程重复运行直到stop被设置，以支持重复launch
                while (!this->_stop) {
                    // 等待lanuch通知
                    std::unique_lock lock(this->_launch_mtx);
                    this->_launch_cv.wait(lock, [this, i]() {
                        // 如果收到停止信号或者开始信号则结束等待
                        return this->_stop || 
                            (this->_launched && this->_kernel_wrapper && !this->_finish_flag[i]);
                    });
                    // 解锁，让其他线程可以运行
                    lock.unlock();

                    if (this->_stop) {
                        break;
                    }
                    // 得到launch通知，执行kernel函数
                    this->_kernel_wrapper(ctxt);

                    // 设置线程当前launch完成标志，避免单个线程重复运行同一个kernel launch
                    this->_finish_flag[i] = true;
                    // 执行完毕通知
                    this->_finish_cv.notify_all();
                }
            }));
        }
    }

    // 禁用复制构造函数
    CPUKernelSimulator(const CPUKernelSimulator&) = delete;

    inline ~CPUKernelSimulator() {
        // 析构时先停止所有线程
        _stop = true;
        _launch_cv.notify_all();
        // 等待所有线程退出
        printf("waiting all threads to exit\n");
        for (std::thread& thread : _threads) {
            thread.join();
        }
        printf("all threads exited, simulator destroied\n");
    }

    /**
     * 启动kernel执行
     * @param kernel kernel函数，函数的第一个参数必须是Context类型
     */
    template<typename F, typename... Args>
    inline void launch(F&& kernel, Args&&... args) {
        std::unique_lock<std::mutex> lock(_launch_mtx);
        if (_launched) {
            throw new std::runtime_error(
                "kernel already launched, please wait for previous launch to finish");
        }

        // 重置所有线程的完成标志
        for (auto i=0; i<_finish_flag.size(); i++) {
            _finish_flag[i] = false;
        }
        // 将除了context之外所有的参数进行绑定
        _kernel_wrapper = std::bind(std::forward<F>(kernel), std::placeholders::_1, 
            std::forward<Args>(args)...);
        _launched = true;
        // 通知所有线程可以开始
        _launch_cv.notify_all();
    }

    /**
     * 等待kernel执行完成
     */
    inline void wait() {
        std::unique_lock<std::mutex> lock(_launch_mtx);
        if (!_launched) {
            return;
        }
        // 等待所有任务完成
        _finish_cv.wait(lock, [this]() {
            return std::all_of(this->_finish_flag.begin(), this->_finish_flag.end(), 
                [](bool flag) {return flag;});
        });
        _kernel_wrapper = nullptr;
        _launched = false;
    }
protected:
    // 标识模拟器当前是否有kernel正在执行
    bool _launched;
    // 终止标识，如果为true，则所有线程执行完任务后退出
    bool _stop;
    // 线程池
    std::vector<std::thread> _threads;
    // 对kernel函数的包装，便于在worker线程中调用，每次launch会设置为当前launch的kernel函数
    std::function<void(Context)> _kernel_wrapper = nullptr;
    std::mutex _launch_mtx;
    std::condition_variable _launch_cv;
    // 线程是否完成当前kernel任务的标志，每个线程一个
    std::vector<bool> _finish_flag;
    std::condition_variable _finish_cv;
};