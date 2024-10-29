#include <stdio.h>
#include <vector>
#include "argparse.hpp"
#include "utils.hpp"


/**
 * 打印kernel中线程的相关id信息
 */
__global__ void print_id() {
    // 按row-major方式计算线程在block中的线性全局坐标
    uint tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    // 按照划分逻辑计算线程的warp id
    uint logic_warpid = tid / warpSize;
    // 从PTX寄存器读取warp id
    uint reg_warpid = warpid();

    printf("block (%d, %d, %d), thread (%d, %d, %d), tid=%d, logical warpid=%d, register warpid=%d\n",
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, tid, logic_warpid, reg_warpid);
}


int main(int argc, char* argv[]) {
    argparse::ArgumentParser parser("warpid");
    std::vector<int> grid_dims{1, 1, 1}, block_dims;

    parser.add_argument("--grid")
        .help("grid形状")
        .nargs(1, 3)
        .store_into(grid_dims);

    parser.add_argument("--block")
        .help("block形状")
        .required()
        .nargs(1, 3)
        .store_into(block_dims);

    parser.parse_args(argc, argv);
    
    dim3 grid{grid_dims[0], grid_dims.size()>1?grid_dims[1]:1, grid_dims.size()>2?grid_dims[2]:1};
    dim3 block{block_dims[0], block_dims.size()>1?block_dims[1]:1, block_dims.size()>2?block_dims[2]:1};

    printf("launch kernel, grid shape=%dx%dx%d, block shape=%dx%dx%d\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    print_id<<<grid, block>>>();
    CUDA_CALL_CHECK(cudaDeviceReset());
    return 0;
}