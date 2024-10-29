#include <chrono>

/**
 * 计时器类
 */
class Timer {
public:
    /**
     * @param start_ 是否开始计时
     */
    explicit Timer(bool start_ = false): _started(start_) {
        if (start_) {
            start();
        }
    }
    ~Timer() {}

    /**
     * 开始计时
     * @param restart 如果之前计时器已经开始计时，是否重新开始；true表示重新开始计时，false表示计时起始时间点不变
     */
    inline void start(bool restart = true) {
        if (restart || !_started) {
            _start_time = std::chrono::steady_clock::now();
        }
        _started = true;
    }

    /**
     * 返回从计时开始到调用本函数时的时间间隔，单位为毫秒
     */
    inline float elapsed_ms() {
        if (!_started) {
            return 0.;
        }
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration dur = end_time - _start_time;
        // 上面的dur的计时单位是纳秒，这里转换成毫秒
        return dur.count()*1.e-6;
    }

private:
    // 计时起始时间点
    std::chrono::steady_clock::time_point _start_time;
    // 计时器是否已经开始计时
    bool _started = false;
};