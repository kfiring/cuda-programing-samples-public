#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include "utils.hpp"

#define Ki (1024.)
#define Mi (Ki*1024)
#define Gi (Mi*1024)
#define K (1000.)
#define M (K*1000.)
#define G (M*1000.)

/**
 * 打印属性名及属性值
 * @param prop_name 属性名
 * @param prop_value 属性值
 */
template<typename T>
void print_prop(const char* prop_name, const T& prop_value) {
    std::cout << std::left << std::setw(40) << prop_name << ": " << prop_value << std::endl;
}


int main(int argc, char* argv[]) {
    // 通过cudaGetDeviceProperties API获取设备属性信息
    cudaDeviceProp prop;
    CUDA_CALL_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::stringstream ss;

    print_prop("Device name", prop.name);

    ss << prop.major << "." << prop.minor;
    print_prop("Compute capability", ss.str());
    ss.str("");

    ss << prop.totalGlobalMem/Mi << "MB";
    print_prop("Total global memory ", ss.str());
    ss.str("");

    ss << prop.clockRate/M << "GHz";
    print_prop("Clock rate", ss.str());
    ss.str("");

    ss << prop.memoryClockRate/M << "GHz";
    print_prop("Memory clock rate", ss.str());
    ss.str("");

    print_prop("Memory bit width", prop.memoryBusWidth);

    ss << prop.memoryClockRate/M*prop.memoryBusWidth/8.*2. << "GB/s";
    print_prop("Global memory bandwidth", ss.str());
    ss.str("");

    print_prop("asynchronous engine count", prop.asyncEngineCount);

    print_prop("Total SM count", prop.multiProcessorCount);

    print_prop("Maximum threads per SM", prop.maxThreadsPerMultiProcessor);

    print_prop("Maximum warps per SM", prop.maxThreadsPerMultiProcessor/prop.warpSize);

    print_prop("Maximum 32-bits registers per SM", prop.regsPerMultiprocessor);

    print_prop("Maximum thread blocks per SM", prop.maxBlocksPerMultiProcessor);

    ss << prop.sharedMemPerMultiprocessor/K << "KB";
    print_prop("Maximum shared memory per SM", ss.str());
    ss.str("");

     ss << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2];
    print_prop("Maximum grid", ss.str());
    ss.str("");

    ss << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2];
    print_prop("Maximum thread block", ss.str());
    ss.str("");

    print_prop("Maximum threads per block", prop.maxThreadsPerBlock);

    ss << prop.sharedMemPerBlock/Ki << "KB";
    print_prop("Maximum shared memory per block", ss.str());
    ss.str("");

    print_prop("Maximum 32-bits registers per block", prop.regsPerBlock);
}