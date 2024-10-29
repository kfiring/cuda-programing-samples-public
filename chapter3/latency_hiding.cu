#include "utils.hpp"
#include "statistic.hpp"
#include "argparse.hpp"
#include "csv.hpp"
#include <string>
#include <iostream>
#include <algorithm>
#include "kernels.hpp"


/**
 * 构造csv文件名：<prefix>.m<mode>.w<nwarps>.g<grid dimensions>.b<block dimensions>.csv
 * @param prefix 文件名前缀
 * @param mode 0表示吞吐量测试，1表示latency测试
 * @param nwarps 总的warp数
 * @param grid_dims grid维度信息
 * @param block_dims thread block维度信息
 * @return csv文件名
 */
std::string makeCSVFilename(const char* prefix, const int& mode, const uint& nwarps, 
                            const std::vector<int>& grid_dims, 
                            const std::vector<int>& block_dims) {
    std::stringstream ss;
    if (prefix && strlen(prefix)) {
        ss << prefix << ".";
    }
    ss << "m" << mode<< ".w" << nwarps << ".g";
    for (auto i=0; i<grid_dims.size(); i++) {
        ss << grid_dims[i];
        if (i < grid_dims.size() - 1) {
            ss << "x";
        }
    }
    ss << ".b";
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
    argparse::ArgumentParser parser("latency_hiding");
    std::vector<int> grid_dims{1, 1}, block_dims{32, 1};
    uint round = 1;
    int mode = 0;
    
    parser.add_argument("--mode", "-m")
        .help("运行模式，1表示做warp latency测量，kernel中会插入计时代码，0表示做吞吐量测试")
        .required()
        .choices(0, 1)
        .store_into(mode);

    parser.add_argument("--grid", "-g")
        .help("grid形状，空格分割的整数，只能1个或2个，1个代表x方向，2个分别代表x,y方向上的thread block数，默认为(1,1)")
        .nargs(1, 2)
        .store_into(grid_dims);


    parser.add_argument("--block", "-b")
        .help("thread block形状，空格分割的整数，只能1个或2个，1个代表x方向，分别代表x,y方向上的线程数，默认为(32,1)")
        .nargs(1, 2)
        .store_into(block_dims);
    
    parser.add_argument("--round", "-r")
        .help("测试轮数，跑指定轮数后，取平均值，默认为1")
        .store_into(round);

    parser.add_argument("-o")
        .help("将运行详细数据写入csv文件，默认不写")
        .flag();
    
    parser.parse_args(argc, argv);
    bool write_detail = parser.get<bool>("-o");

    if (grid_dims.size() < 2) {
        grid_dims.emplace_back(1);
    }
    if (block_dims.size() < 2) {
        block_dims.emplace_back(1);
    }

    assert(std::all_of(grid_dims.begin(), grid_dims.end(), [](int d){return d>0;}));
    assert(std::all_of(block_dims.begin(), block_dims.end(), [](int d){return d>0;}));
    assert(round > 0);

    cudaDeviceProp cuda_props;
    CUDA_CALL_CHECK(cudaGetDeviceProperties(&cuda_props, 0));

    // 根据grid及thread block形状计算矩阵大小：m行，n列
    uint m = grid_dims[1]*block_dims[1];
    uint n = grid_dims[0]*block_dims[0];
    printf("--> matrix shape: %dx%d\n", m, n);

    // 分配矩阵内存
    uint nelem = m * n;
    size_t nelem_bytes = nelem * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CALL_CHECK(cudaMalloc(&d_A, nelem_bytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_B, nelem_bytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_C, nelem_bytes));

    // 分配时延记录内存
    uint nwarps_per_block = (block_dims[0] * block_dims[1] + cuda_props.warpSize -1 ) / cuda_props.warpSize;
    uint nwarps = nwarps_per_block * grid_dims[0] * grid_dims[1];
    printf("--> # warps: %d, # warps per block: %d, #warps per SM: %d\n", nwarps, nwarps_per_block,
        nwarps/cuda_props.multiProcessorCount);
    size_t records_bytes = nwarps * sizeof(WarpRunRecord);
    WarpRunRecord *d_records = nullptr, *h_records = nullptr;
    if (mode == 1) {
        CUDA_CALL_CHECK(cudaMalloc(&d_records, records_bytes));
        h_records = (WarpRunRecord*)malloc(records_bytes);
    }

    
    dim3 grid{grid_dims[0], grid_dims[1]}, block{block_dims[0], block_dims[1]};
    printf("--> grid shape: %dx%dx%d, thread block shape: %dx%dx%d\n", 
        grid.x, grid.y, grid.z, block.x, block.y, block.z);

    // warming up一下
    if (mode == 1) {
        matrixadd2d<1><<<grid, block, 0>>>(d_A, d_B, d_C, m, n, d_records);
    } else {
        matrixadd2d<0><<<grid, block, 0>>>(d_A, d_B, d_C, m, n, d_records);
    }
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    printf("--> warming up finished\n");

    std::unique_ptr<csv::CSVWriter<std::ofstream>> p_csv_writer;
    if (write_detail) {
        p_csv_writer = std::make_unique<csv::CSVWriter<std::ofstream>>(
            makeCSVFilename("warpdetail", mode, nwarps, grid_dims, block_dims));
        // 写csv文件头
        std::vector<std::string> csv_cols{"round_id", "kernel_time(ms)", "warp_throughput(Warps/s)", 
            "memory_throughput(GB/s)"};
        if (mode == 1) {
            csv_cols.emplace_back("warp_latency_mean(cycles)");
            csv_cols.emplace_back("warp_latency_max(cycles)");
            csv_cols.emplace_back("warp_latency_min(cycles)");
            csv_cols.emplace_back("warp_latency_std(cycles)");
        }
        *p_csv_writer << csv_cols;        
    }

    // 全局统计：对所有轮次的均值进行统计
    // g_cycle_stats: 统计warp latency
    // g_time_stats: 统计kernel耗时
    SimpleStats<float> g_cycle_stats, g_time_stats;
    for (auto r=0; r<round; r++) {
        if (mode == 1) {
            // 每一轮之前记录及计算结果归0
            CUDA_CALL_CHECK(cudaMemset(d_records, 0, records_bytes));
            CUDA_CALL_CHECK(cudaMemset(d_C, 0, nelem_bytes));
            memset(h_records, 0, records_bytes);
        }

        // 获取kernel耗时
        float time = 0;
        if (mode == 1) {
            time = TIME_KERNEL(matrixadd2d<1>, grid, block, 0, 0, d_A, d_B, d_C, m, n, d_records);
        } else {
            time = TIME_KERNEL(matrixadd2d<0>, grid, block, 0, 0, d_A, d_B, d_C, m, n, d_records);
        }
        g_time_stats.append(time);

        // 计算整体有效memory吞吐：2个读size + 1个写size，GB/s
        float mem_thr = 3*nelem_bytes*1000/time/(1024*1024*1024);
        // 计算整体warp吞吐：warps/s
        float warp_thr = nwarps*1000/time;

        if (mode == 1) {
            // 记录传回host
            CUDA_CALL_CHECK(cudaMemcpy(h_records, d_records, records_bytes, cudaMemcpyDeviceToHost));
            // 每一轮的统计：每一轮内不同warp的latency取平均值
            SimpleStats<float> cycle_stats;
            cycle_stats.append_many<WarpRunRecord>(h_records, nwarps, [](size_t i, const WarpRunRecord& r){
                // 计算单个warp的耗时
                return (float)(r.cycle_cost);
            });
            // 单轮平局值记入全局统计中
            g_cycle_stats.append(cycle_stats.mean());
            if (write_detail) {
                // 明细写入csv
                *p_csv_writer << std::make_tuple(r, time, warp_thr, mem_thr, cycle_stats.mean(), 
                    cycle_stats.max(), cycle_stats.min(), cycle_stats.std());
            }
        } else if(write_detail) {
            // 明细写入csv
            *p_csv_writer << std::make_tuple(r, time, warp_thr, mem_thr);
        }
    }
    // 打印
    printf("--> %d warps kernel time (ms): max=%f, min=%f, mean=%f, std=%f\n", nwarps,
        g_time_stats.max(), g_time_stats.min(), g_time_stats.mean(), g_time_stats.std());
    printf("--> mean warp througput: %f warps/s\n", nwarps*1000/g_time_stats.mean());
    printf("--> mean memory throughput: %f GB/s\n", 3*nelem_bytes*1000/(g_time_stats.mean()*1024*1024*1024));
    if (mode == 1) {
        printf("--> warp latency over %d rounds: max=%f, min=%f, mean=%f, std=%f\n", 
            g_cycle_stats.count(), g_cycle_stats.max(), g_cycle_stats.min(), g_cycle_stats.mean(), 
            g_cycle_stats.std());
    } 

    // 释放资源
    CUDA_CALL_CHECK(cudaFree(d_A));
    CUDA_CALL_CHECK(cudaFree(d_B));
    CUDA_CALL_CHECK(cudaFree(d_C));
    CUDA_CALL_CHECK(cudaFree(d_records));
    CUDA_CALL_CHECK(cudaDeviceReset());
    free(h_records);
    return 0;
}