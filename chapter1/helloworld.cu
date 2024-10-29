#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("thread[%d]: hello world from GPU!\n", threadIdx.x);
}


int main(void) {
    printf("hello world from CPU!\n");
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}