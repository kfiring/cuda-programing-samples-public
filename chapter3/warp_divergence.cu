
#include <stdio.h>
#include "utils.hpp"


__global__ void print_clock() {
    // 按row-major方式计算线程在block中的线性全局坐标
    uint tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    // 按照划分逻辑计算线程的warp id
    uint warpid = tid / warpSize;
    // laneid是线程在warp内的id
    uint laneid = tid % warpSize;

    // 读取时钟周期计数
    ulong clock = cuda_clock();
    // 分支前打印clock
    printf("pre divergence, tid=%d, warpid=%d, laneid=%d, clock=%lu\n", tid, warpid, laneid, clock);
    // 将warp中线程分成两部分，造成divergence，每个分支内打印clock
    if (laneid < warpSize/2) {
        clock = cuda_clock();
        printf("1st half, tid=%d, warpid=%d, laneid=%d, clock=%lu\n", tid, warpid, laneid, clock);
    } else {
        clock = cuda_clock();
        printf("2nd half, tid=%d, warpid=%d, laneid=%d, clock=%lu\n", tid, warpid, laneid, clock);
    }
    // 分之后打印clock
    clock = cuda_clock();
    printf("after divergence, tid=%d, warpid=%d, laneid=%d, clock=%lu\n", tid, warpid, laneid, clock);
}


int main(int argc, char* argv[]) {
    print_clock<<<1, 32>>>();
    CUDA_CALL_CHECK(cudaDeviceReset());
    return 0;
}