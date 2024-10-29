#include <stdio.h>

__global__ void printThreadInfo() {
    // 计算线程全局坐标
    uint ix = blockIdx.x*blockDim.x + threadIdx.x;
    uint iy = blockIdx.y*blockDim.y + threadIdx.y;
    uint iz = blockIdx.z*blockDim.z + threadIdx.z;
    printf("->grid shape: %dx%dx%d, block shape: %dx%dx%d, block coordinates: (%d, %d, %d), "
        "thread local coordinates: (%d, %d, %d), thread global coordinates: (%d, %d, %d)\n", 
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, 
        ix, iy, iz);
}

int main(void) {
    // 这里我们采用了1维grid，3维thread block
    dim3 grid(2);
    dim3 block(3, 2, 2);
    printThreadInfo<<<grid, block>>>();
    cudaDeviceReset();
    return 0;
}