#include <stdlib.h>
#include <stdio.h>
#include "utils.hpp"
#include "timer.hpp"
#include "cpu_kernel_simulator.hpp"


/**
 * CPU版本向量加和：C = A + B
 * @param A 向量A的首地址
 * @param B 向量B的首地址
 * @param C 向量C的首地址
 * @param N 向量维度大小
 */
void sumVectorsOnHost(float *A, float *B, float *C, const int N) {
    for (int i=0; i<N; i++) {
        C[i] = A[i] + B[i];
    }
}

/**
 * 多线程CPU模拟器版本向量加和：C = A + B
 * @param ctxt 模拟器线程上下文
 * @param A 向量A的首地址
 * @param B 向量B的首地址
 * @param C 向量C的首地址
 * @param N 向量维度大小
 */
void sumVectorsOnHostKernel(CPUKernelSimulator::Context ctxt, float *A, float *B, float *C, const int N) {
    // 计算每个线程需要处理的数据条数
    int batch_size = (N+ctxt.total_thread_count-1) / ctxt.total_thread_count;
    // 根据数据条数计算当前线程处理数据的索引起止范围
    int i=batch_size*ctxt.thread_idx, j=min(i+batch_size, N);
    // 计算
    for (; i<j; i++) {
        C[i] = A[i] + B[i];
    }
}

/**
 * GPU版本向量加和：C = A + B
 * @param A 向量A的首地址
 * @param B 向量B的首地址
 * @param C 向量C的首地址
 * @param N 向量维度大小
 */
__global__ void sumVectorsOnGPU(float *A, float *B, float *C, const int N) {
    // 计算线程全局索引，对应到向量中的元素索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // 这里加一个判断，避免索引越界
        C[i] = A[i] + B[i];
    }
}


int main(int argc, char* argv[]) {
    // 通过命令行参数获取thread block大小（1维）,默认512
    uint block_x = 512;
    if (argc > 1) {
        block_x = atoi(argv[1]);
    }

    // 定义向量维度大小，这里我们设置为1^27
    constexpr int N = 1 << 27;
    // 计算单个向量所需字节数
    constexpr size_t nbytes = N * sizeof(float); 
    // 为向量分配host内存
    float *h_A = (float *)malloc(nbytes);
    float *h_B = (float *)malloc(nbytes);
    float *host_C = (float *)malloc(nbytes);
    float *gpu_C = (float *)malloc(nbytes);

    // 初始化向量A、B的内容
    randInitialData(h_A, N);
    randInitialData(h_B, N);
    // 将向量C的内容置0
    memset(host_C, 0, N);
    memset(gpu_C, 0, N);

    CPUKernelSimulator cpu_kernel_simulater(128);
    // 对CPU版本的函数调用进行预热
    printf("--> start warming up on CPU...\n");
    Timer timer(true);
    // sumVectorsOnHost(h_A, h_B, host_C, N);
    cpu_kernel_simulater.launch(sumVectorsOnHostKernel, h_A, h_B, host_C, N);
    cpu_kernel_simulater.wait();
    printf("--> CPU warming up cost %0.6fms\n", timer.elapsed_ms());

    // 执行CPU版本的函数
    timer.start();
    // sumVectorsOnHost(h_A, h_B, host_C, N);
    cpu_kernel_simulater.launch(sumVectorsOnHostKernel, h_A, h_B, host_C, N);
    cpu_kernel_simulater.wait();
    printf("--> CPU calculation cost %0.6fms\n", timer.elapsed_ms());

    // 为向量分配device内存
    float *d_A, *d_B, *d_C;
    CUDA_CALL_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_B, nbytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_C, nbytes));

    // 将向量A，B内容从host拷贝到GPU
    CUDA_CALL_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice));

    // 计算thread block的宽度
    dim3 block = dim3(block_x);
    // 计算grid的宽度，如果N不能被block_x整除，则最后一个block中会有一些线程空闲
    dim3 grid = dim3((N + block_x -1)/block_x);
    printf("--> launching kernel: grid shape: %dx%dx%d, block shape: %dx%dx%d\n", 
        grid.x, grid.y, grid.z, block.x, block.y, block.z);

    printf("--> start CUDA kernel warming up...\n");
    // 对CUDA kernel进行预热
    timer.start();
    sumVectorsOnGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    // 执行一次device级的同步，等待kernel执行完成（这个操作比较重，一般情况下不轻易使用，这个是为了计时示例）
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    printf("--> CUDA kernel warming up cost %0.6fms\n", timer.elapsed_ms());
    
    // CUDA kernel launch
    timer.start();
    sumVectorsOnGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    printf("--> GPU calculation cost %0.6fms\n", timer.elapsed_ms());

    // 将GPU的计算结果拷贝回host
    CUDA_CALL_CHECK(cudaMemcpy(gpu_C, d_C, nbytes, cudaMemcpyDeviceToHost));

    // 校验CPU和GPU计算结果的一致性
    checkResult(host_C, gpu_C, N);

    // 释放device内存
    CUDA_CALL_CHECK(cudaFree(d_A));
    CUDA_CALL_CHECK(cudaFree(d_B));
    CUDA_CALL_CHECK(cudaFree(d_C));
    CUDA_CALL_CHECK(cudaDeviceReset());

    // 释放host内存
    free(h_A);
    free(h_B);
    free(host_C);
    free(gpu_C);
    return 0;
}