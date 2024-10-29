#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include "argparse.hpp"
#include "utils.hpp"
#include "kernels.hpp"
#include "statistic.hpp"
#include "csv.hpp"


/**
 * 构造csv文件名：<prefix>.<m>x<n>.g<grid dimensions>.b<block dimensions>.csv
 * @param prefix 文件名前缀
 * @param m 矩阵行数
 * @param n 矩阵列数
 * @param block_dims thread block维度信息
 * @return csv文件名
 */
std::string makeCSVFilename(const char* prefix, const uint& m, const uint& n,
                            const std::vector<int>& block_dims) {
    std::stringstream ss;
    if (prefix && strlen(prefix)) {
        ss << prefix << ".";
    }
    ss << m << "x" << n << ".b";
    for (auto i=0; i<block_dims.size(); i++) {
        ss << block_dims[i];
        if (i < block_dims.size() - 1) {
            ss << "x";
        }
    }
    ss << ".csv";
    return ss.str();
}


int main(int argc, char* argv[]) {
    // 解析参数：grid，thread block的形状信息、矩阵的长宽信息
    argparse::ArgumentParser parser("matrixadd2d");
    std::vector<int> block_dims{32, 1};
    uint m = 0, n = 0;
    uint round = 1;

    parser.add_argument("--block", "-b")
        .help("thread block形状，空格分割的整数，只能1个或2个，1个代表x方向，分别代表x,y方向上的线程数")
        .nargs(1, 2)
        .store_into(block_dims);

    parser.add_argument("-m")
        .help("矩阵行数")
        .required()
        .store_into(m);
    
    parser.add_argument("-n")
        .help("矩阵列数")
        .required()
        .store_into(n);
    
    parser.add_argument("--round", "-r")
        .help("测试轮数，跑指定轮数后，取平均值，默认为1")
        .store_into(round);

    parser.add_argument("-o")
        .help("将运行详细数据写入csv文件，默认不写")
        .flag();
    
    parser.parse_args(argc, argv);
    bool write_detail = parser.get<bool>("-o");

    assert(std::all_of(block_dims.begin(), block_dims.end(), [](int d){return d>0;}));
    assert(m > 0 && n > 0);
    assert(round > 0);

    if (block_dims.size() < 2) {
        block_dims.emplace_back(1);
    }


    printf("--> matrix shape: %dx%d\n", m, n);

    // 分配矩阵内存
    uint nelem = m * n;
    size_t nelem_bytes = nelem * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CALL_CHECK(cudaMalloc(&d_A, nelem_bytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_B, nelem_bytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_C, nelem_bytes));
    

    // 计算设置grid，thread block形状
    dim3 block{block_dims[0], block_dims[1]}, grid;
    grid.x = (n + block.x - 1) / block.x;
    grid.y = (m + block.y - 1) / block.y;

    printf("--> grid shape: %dx%dx%d, block shape: %dx%dx%d\n", grid.x, grid.y, grid.z, 
        block.x, block.y, block.z);
    // 对CUDA kernel进行预热
    matrixadd2d<0><<<grid, block>>>(d_A, d_B, d_C, m, n);
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    printf("--> CUDA kernel warming up finished\n");

    std::unique_ptr<csv::CSVWriter<std::ofstream>> p_csv_writer;
    if (write_detail) {
        // 写csv文件头
        p_csv_writer = std::make_unique<csv::CSVWriter<std::ofstream>>(
            makeCSVFilename("matrixadd2d-time", m, n, block_dims));
        *p_csv_writer << std::vector<std::string>{"round", "time(ms)"};
    }

    SimpleStats<float> stats;
    for (auto r=0; r<round; r++) {
        float time = TIME_KERNEL(matrixadd2d<0>, grid, block, 0, 0, d_A, d_B, d_C, m, n);
        stats.append(time);
        if (write_detail) {
            // 耗时记入csv文件
            *p_csv_writer << std::make_tuple(r, time);
        }
    }

    printf("--> kernel time over %d rounds (ms): max=%f, min=%f, mean=%f, std=%f\n", 
        stats.count(), stats.max(), stats.min(), stats.mean(), stats.std());

    // 释放device内存
    CUDA_CALL_CHECK(cudaFree(d_A));
    CUDA_CALL_CHECK(cudaFree(d_B));
    CUDA_CALL_CHECK(cudaFree(d_C));
    CUDA_CALL_CHECK(cudaDeviceReset());
    return 0;
}