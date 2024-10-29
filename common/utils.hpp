#pragma once
#include <iostream>
#include <sstream>
#include <chrono>
#include <time.h>
#include <cuda.h>

/**
 * 对CUDA API调用结果进行检查，如果调用出错，则抛出异常
 */
#define CUDA_CALL_CHECK(expr)                                             \
    do {                                                                  \
        cudaError_t status = (expr);                                      \
        if (status != cudaSuccess) {                                      \
            std::stringstream ss;                                         \
            ss << "cuda call '" << #expr << "' error: "                   \
                << status << ", msg: " << cudaGetErrorString(status);     \
            std::cerr << ss.str() << std::endl;                           \
            throw std::runtime_error(ss.str());                           \
        }                                                                 \
    } while (0)


/**
 * 将总数`denom`平均分成`num`份，每份之间差别不超过1，并返回第`i`份的大小
 * 例如：100分成6份，则结果是[17,17,17,17,16,16]
 */
#define EVEN_DIVIDE(denom, num, i) ((denom)/(num) + uint((i) < (denom)%(num)))


/**
 * 调用CUDA kernel并进行计时
 * @param kernel_func 要调用的核函数
 * @param grid grid配置
 * @param block block配置
 * @param sm_size shared memory大小
 * @param stream 使用的stream句柄
 * @param args... 传递给核函数的参数
 * @return float 核函数的耗时，单位为毫秒
 * 
 */
#define TIME_KERNEL(kernel_func, grid, block, sm_size, stream, args...)     \
    ({                                                                      \
        cudaEvent_t _start_evt, _end_evt;                                   \
        CUDA_CALL_CHECK(cudaEventCreate(&_start_evt));                      \
        CUDA_CALL_CHECK(cudaEventCreate(&_end_evt));                        \
        CUDA_CALL_CHECK(cudaEventRecord(_start_evt));                       \
        kernel_func<<<grid, block, sm_size, stream>>>(args);                \
        CUDA_CALL_CHECK(cudaEventRecord(_end_evt));                         \
        CUDA_CALL_CHECK(cudaEventSynchronize(_end_evt));                    \
        float time = 0;                                                     \
        CUDA_CALL_CHECK(cudaEventElapsedTime(&time, _start_evt, _end_evt)); \
        CUDA_CALL_CHECK(cudaEventDestroy(_start_evt));                      \
        CUDA_CALL_CHECK(cudaEventDestroy(_end_evt));                        \
        time;                                                               \
    })


/**
 * 使用随机数填充数组
 * @param p 待填充数组的首地址
 * @param N 填充长度
 */
void randInitialData(float *p, const int N) {
    srand((unsigned int)time(nullptr));
    for (int i=0; i<N; i++) {
        // 使用0~1内的随机数填充
        p[i] = ((float)rand())/RAND_MAX;
    }
}

/**
 * 校验host和GPU的计算结果是否一致，如果不匹配则打印错误消息
 * @param host_C CPU版本的计算结果首地址
 * @param gpu_C GPU版本的计算结果首地址
 * @param N 数组长度
 */
void checkResult(float *host_C, float *gpu_C, const int N) {
    // 单个值最大允许误差
    double epsilon = 1.0e-8;
    // 循环校验向量中的每个元素
    for (int i=0; i<N; i++) {
        if (abs(host_C[i] - gpu_C[i]) > epsilon) {
            printf("ERROR: result miss-match at %dth element, host[%d]=%f, gpu[%d]=%f\n", 
                i, i, host_C[i], i, gpu_C[i]);
            break;
        }
    }
}


/**
 * 从PTX specicial register中获取warpid，
 * 参考：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers
 * 注意：
 * 1. %warpid寄存器值在block内是唯一的，不同block中的warpid是独立的。这与warpid计算逻辑是一样的。
 * 2. %warpid寄存器值与逻辑warpid有一点区别，就是当发生线程抢占重新调度时，一个线程的%warpid可能会发生变化，
 *    没有发生线程的重新调度时，%warpid与warpid计算逻辑是一样的。因此这个函数我们只用于实验验证，实际应用中
 *    如果需要用到warpid值，还是应该用threadIdx去计算。
 */
inline __device__ uint warpid() {
    uint ret = -1;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}


/**
 * 从PTX specicial register中获取时钟计数
 */
inline __device__ unsigned long cuda_clock() {
    unsigned long ret = 0;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(ret));
    return ret;
}


/**
 * 从PTX specicial register中获取调用线程所在SM的id
 */
inline __device__ uint smid() {
    uint ret = -1;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

