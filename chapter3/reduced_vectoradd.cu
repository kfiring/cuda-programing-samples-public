#include "utils.hpp"
#include "kernels.hpp"
#include "argparse.hpp"
#include "statistic.hpp"
#include <assert.h>
#include <iostream>


int main(int argc, char *argv[]) {
    argparse::ArgumentParser parser("reduced_vectoradd");
    uint nelem_power = 24, block_x_power = 9;
    uint round = 1;
    uint method = 0, block_unroll_factor = 2;

    parser.add_argument("-n")
        .help("向量元素个数，为保证个数是2的幂次方，该参数为次方数，即最终元素个数为2^n个，最大值31，默认为24，即2^24个")
        .store_into(nelem_power);

    parser.add_argument("--block", "-b")
        .help("block的x维大小，为保证个数是2的幂次方，该参数为次方数，即大小为2^b，最大值10，默认为9，即2^9=512")
        .store_into(block_x_power);
      
    parser.add_argument("--round", "-r")
        .help("测试轮数，跑指定轮数后，取平均值，默认为1")
        .store_into(round);

    parser.add_argument("--method", "-m")
        .help("指定使用哪种规约算法，1表示相邻聚合；2表示连续线程+相邻聚合；3表示连续线程+交叉聚合；"
            "4表示在3的基础上加上跨block的聚合；5表示在4的基础上加上最后一个warp的unrolling；"
            "6表示在5的基础上加上block内所有线程的unrolling；7表示采用递归方式，每个block启动一个子grid；"
            "8表示采用递归方式，每次只启动一个子grid")
        .choices(1,2,3,4,5,6,7,8)
        .required()
        .store_into(method);
    
    parser.add_argument("--buf")
        .help("block的unrolling factor，即几个相邻block的数据先聚合，仅对method为4,5,6时有效，默认为2")
        .choices(2,4,8)
        .store_into(block_unroll_factor);
    
    
    parser.parse_args(argc, argv);
    assert(nelem_power > 0 && nelem_power <= 31);
    assert(block_x_power > 0 && block_x_power <= 10);
    assert(round > 0);

    uint nelem = 1<<nelem_power;
    uint block_x = 1<<block_x_power;
    uint grid_x = (nelem + block_x - 1)/block_x;

    if (method == 7) {
        // method 7中，因为每个block都会启动子grid，一共启动block_x_power层，如果数据多可能会启动大量grid导致
        // pending lanunch超限，所以这里修改一下上限值
        CUDA_CALL_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, grid_x*block_x_power));
    }

    // 打印一些限制信息
    for (auto l : std::vector<cudaLimit>{cudaLimitStackSize, cudaLimitPrintfFifoSize, 
        cudaLimitMallocHeapSize, cudaLimitDevRuntimeSyncDepth, cudaLimitDevRuntimePendingLaunchCount,
        cudaLimitMaxL2FetchGranularity, cudaLimitPersistingL2CacheSize}) {
        std::string str = "";
        switch (l)
        {
        case cudaLimitStackSize:
            str = "cudaLimitStackSize";
            break;
        case cudaLimitPrintfFifoSize:
            str = "cudaLimitPrintfFifoSize";
            break;
        case cudaLimitMallocHeapSize:
            str = "cudaLimitMallocHeapSize";
            break;
        case cudaLimitDevRuntimeSyncDepth:
            str = "cudaLimitDevRuntimeSyncDepth";
            break;
        case cudaLimitDevRuntimePendingLaunchCount:
            str = "cudaLimitDevRuntimePendingLaunchCount";
            break;
        case cudaLimitMaxL2FetchGranularity:
            str = "cudaLimitMaxL2FetchGranularity";
            break;
        case cudaLimitPersistingL2CacheSize:
            str = "cudaLimitPersistingL2CacheSize";
            break;
        default:
            break;
        }
        size_t val = 0;
        cudaError_t err = cudaDeviceGetLimit(&val, l);
        std::cout << str << ":";
        if (err == cudaSuccess) {
            std::cout << val;
        } else {
            std::cout << "not support(" << cudaGetErrorString(err) << ")";
        }
        std::cout << std::endl;
    }

    printf("--> using method %d\n", method);
    printf("--> vector size: %d\n", nelem);
    if (method == 6 && block_x != 512) {
        // 简化起见，当使用method 6时，强制要求block size必须是512
        throw std::runtime_error("block must be 9 when using method 6");
    }
    if (method >= 4 && method <= 6) {
        // 如果使用了block unrolling，则block数相应降低
        printf("--> block urolling factor: %d\n", block_unroll_factor);
        grid_x /= block_unroll_factor;
    }
    if (method == 8) {
        // method 8中所需block数减半
        grid_x /= 2;
    }

    // 在主机上分配向量及结果矩阵
    float *h_i_data = new float[nelem];
    float *h_o_data = new float[grid_x];
    // 随机初始化输入向量
    randInitialData(h_i_data, nelem);

    // 在GPU上分配向量及结果矩阵
    size_t ni_bytes = nelem * sizeof(float);
    size_t no_bytes = grid_x * sizeof(float);
    float *d_i_data=nullptr, *d_o_data=nullptr;
    CUDA_CALL_CHECK(cudaMalloc(&d_i_data, ni_bytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_o_data, no_bytes));
    
    // 将输入向量从host拷贝到GPU
    CUDA_CALL_CHECK(cudaMemcpy(d_i_data, h_i_data, ni_bytes, cudaMemcpyHostToDevice));

    dim3 grid(grid_x), block(block_x);
    printf("--> grid shape: %dx%dx%d, block shape: %dx%dx%d\n", grid.x, grid.y, grid.z, 
        block.x, block.y, block.z);
    // warm up
    float *temp_i_data = nullptr;
    CUDA_CALL_CHECK(cudaMalloc(&temp_i_data, ni_bytes));
    switch (method)
    {
    case 1:
        reducedVectorAdd1<<<grid, block>>>(temp_i_data, d_o_data, nelem);
        break;
    case 2:
        reducedVectorAdd2<<<grid, block>>>(temp_i_data, d_o_data, nelem);
        break;
    case 3:
        reducedVectorAdd3<<<grid, block>>>(temp_i_data, d_o_data, nelem);
        break;
    case 4:
        switch (block_unroll_factor)
        {
        case 2:
            reducedVectorAddUnrolling1<2><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        case 4:
            reducedVectorAddUnrolling1<4><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        case 8:
            reducedVectorAddUnrolling1<8><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        default:
            break;
        }
        break;
    case 5:
        switch (block_unroll_factor)
        {
        case 2:
            reducedVectorAddUnrolling2<2><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        case 4:
            reducedVectorAddUnrolling2<4><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        case 8:
            reducedVectorAddUnrolling2<8><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        default:
            break;
        }
        break;
    case 6:
        switch (block_unroll_factor)
        {
        case 2:
            reducedVectorAddUnrolling3<2><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        case 4:
            reducedVectorAddUnrolling3<4><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        case 8:
            reducedVectorAddUnrolling3<8><<<grid, block>>>(temp_i_data, d_o_data, nelem);
            break;
        default:
            break;
        }
        break;
    case 7:
        reducedVectorAddRecursive1<<<grid, block>>>(temp_i_data, d_o_data, nelem);
        break;
    case 8:
        reducedVectorAddRecursive2<<<grid, block>>>(temp_i_data, d_o_data, nelem, block_x);
        break;
    default:
        break;
    }
    CUDA_CALL_CHECK(cudaDeviceSynchronize());
    CUDA_CALL_CHECK(cudaFree(temp_i_data));
    printf("--> warm up finished\n");

    CUDA_CALL_CHECK(cudaMemset(d_o_data, 0, no_bytes));
    SimpleStats<float> stats;
    for (auto r=0; r<round; r++) {
        // 启动kernel并计时
        float time = 0;
        switch (method)
        {
        case 1:
            time = TIME_KERNEL(reducedVectorAdd1, grid, block, 0, 0, d_i_data, d_o_data, nelem);
            break;
        case 2:
            time = TIME_KERNEL(reducedVectorAdd2, grid, block, 0, 0, d_i_data, d_o_data, nelem);
            break;
        case 3:
            time = TIME_KERNEL(reducedVectorAdd3, grid, block, 0, 0, d_i_data, d_o_data, nelem);
            break;
        case 4:
            switch (block_unroll_factor)
            {
            case 2:
                time = TIME_KERNEL(reducedVectorAddUnrolling1<2>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            case 4:
                time = TIME_KERNEL(reducedVectorAddUnrolling1<4>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            case 8:
                time = TIME_KERNEL(reducedVectorAddUnrolling1<8>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            default:
                break;
            }
            break;
        case 5:
            switch (block_unroll_factor)
            {
            case 2:
                time = TIME_KERNEL(reducedVectorAddUnrolling2<2>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            case 4:
                time = TIME_KERNEL(reducedVectorAddUnrolling2<4>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            case 8:
                time = TIME_KERNEL(reducedVectorAddUnrolling2<8>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            default:
                break;
            }
            break;
        case 6:
            switch (block_unroll_factor)
            {
            case 2:
                time = TIME_KERNEL(reducedVectorAddUnrolling3<2>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            case 4:
                time = TIME_KERNEL(reducedVectorAddUnrolling3<4>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            case 8:
                time = TIME_KERNEL(reducedVectorAddUnrolling3<8>, grid, block, 0, 0, d_i_data, d_o_data, nelem);
                break;
            default:
                break;
            }
            break;
        case 7:
            time = TIME_KERNEL(reducedVectorAddRecursive1, grid, block, 0, 0, d_i_data, d_o_data, nelem);
            break;
        case 8:
            time = TIME_KERNEL(reducedVectorAddRecursive2, grid, block, 0, 0, d_i_data, d_o_data, nelem, block_x);
            break;
        default:
            break;
        }
        stats.append(time);
    }

    if (round == 1) {
        // 只有在round为1时才执行最终聚合，因为规约过程是in place的，所以如果跑了多轮，结果就肯定已经不对了
        // 结果拷贝回host，进行最终的聚合
        CUDA_CALL_CHECK(cudaMemcpy(h_o_data, d_o_data, no_bytes, cudaMemcpyDeviceToHost));
        // 执行最终聚合
        double ret = 0;
        for (auto i=0; i<grid_x; i++) {
            ret += h_o_data[i];
        }

        // CPU计算
        double gt = 0;
        for (auto i=0; i<nelem; i++) {
            gt += h_i_data[i];
        }
        printf("--> GPU result %f, CPU result %f\n", ret, gt);
    }

    printf("--> kernel time over %d rounds (ms): max=%f, min=%f, mean=%f, std=%f\n", 
        stats.count(), stats.max(), stats.min(), stats.mean(), stats.std());

    // 释放资源
    CUDA_CALL_CHECK(cudaFree(d_i_data));
    CUDA_CALL_CHECK(cudaFree(d_o_data));
    CUDA_CALL_CHECK(cudaDeviceReset());
    delete [] h_i_data;
    delete [] h_o_data;
    return 0;
}