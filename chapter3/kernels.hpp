#include "utils.hpp"

#ifdef UNROLL_VOLATILE
    #define VOLATILE volatile
#else
    #define VOLATILE
#endif


/**
 * warp运行记录结构体
 */
struct WarpRunRecord {
    // // 起始cycle读数
    // uint64_t cycle_start;
    // // 结束cycle读数
    // uint64_t cycle_end;
    uint cycle_cost;
};



/**
 * 矩阵相加Warp Latency测量，每个线程只处理一个数据
 * 函数通过模板变量MODE区分是否插入计时代码，0表示正常运行，1表示插入计时代码
 * @param A A矩阵首地址
 * @param B B矩阵首地址
 * @param C C矩阵首地址
 * @param m 矩阵的行数
 * @param n 矩阵的额列数
 * @param p_records 运行结果记录数据首地址指针
 */
template<uint MODE>
__global__ void matrixadd2d(const float *A, const float *B, float *C, 
                            const uint m, const uint n, 
                            WarpRunRecord* p_records=nullptr) {
    uint64_t cycle_start = 0, cycle_end = 0;
    if (MODE == 1) {
        // 计时开始
        cycle_start = cuda_clock();
    }

    //------------------------
    // kernel计算逻辑部分
    // 1. 计算线程的全局坐标（i_t, j_t）
    uint i_t = blockIdx.x * blockDim.x + threadIdx.x;
    uint j_t = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_t < n && j_t < m) {
        // 2. 数据逻辑坐标转换为物理坐标
        uint i_p = j_t * n + i_t;
        // 3. 执行计算
        C[i_p] = A[i_p] + B[i_p];
    }
    //------------------------

    if (MODE == 1) {
        // 计时结束
        cycle_end = cuda_clock();
        // 计算线程在block中的线性坐标
        uint b_tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
        // 计算线程在block内的warp id
        uint b_warpid = b_tid / warpSize;
        // 计算线程lane id，即线程在warp内的id
        uint laneid = b_tid % warpSize;
        if (laneid == 0) {
            // 由warp内的第一个线程负责记录
            // 计算线程所在warp在全局中的warp id，以便计算记录偏移量
            // 先计算线程全局坐标:
            //  block_size: 每个block内的线程数
            //  warps_per_block: 每个block内的warp数
            //  bid: block在grid中的线性坐标
            uint block_size = blockDim.x * blockDim.y * blockDim.z;
            uint warps_per_block = (block_size + warpSize -1)/warpSize;
            uint bid = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +  blockIdx.x;
            // 全局warp id
            uint g_warpid = bid * warps_per_block + b_warpid;
            // 记录
            // p_records[g_warpid].smid = smid();
            // p_records[g_warpid].cycle_start = cycle_start;
            // p_records[g_warpid].cycle_end = cycle_end;
            p_records[g_warpid].cycle_cost = cycle_end - cycle_start;
        }
    }
}


/**
 * 向量规约求和，每个block聚合一部分数据，block的聚合结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 1. 每次聚合相邻stride的数据
 * 2. 执行聚合操作的线程为左侧stride对应的线程
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
__global__ void reducedVectorAdd1(float *i_data, float *o_data, const uint n) {
    // 计算当前block处理数据块的首地址
    i_data += blockIdx.x * blockDim.x;
    
    uint tid = threadIdx.x;
    // stride代表含义：
    // 1. 数据角度：已经聚合完成连续数据块宽度，每个stride的聚合数据点是stride的首元素。例如初始stride为1表示每个元素都聚合了自身，
    //    stride为2时表示聚合了右边一个邻居，聚合点在index为偶数位置，以此类推。
    // 2. 线程角度：当前迭代要聚合相邻两个stride，聚合数据点是左边stride的首元素，执行聚合的线程与聚合数据点位置一样，
    //    即tid % (2 * stride) == 0的点
    for (size_t stride=1; stride + tid < blockDim.x; stride<<=1) {
        if (tid % (2*stride) == 0) {
            // 左侧stride的线程聚合右侧stride
            i_data[tid] += i_data[tid + stride];
        }
        // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证下次迭代之前，上次聚合结果已经写入
        __syncthreads();
    }
    // 最后由0号线程将所在block规约结果写入输出内存
    if (tid == 0) {
        o_data[blockIdx.x] = i_data[0];
    }
}


/**
 * 向量规约求和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 1. 每次聚合相邻stride的数据
 * 2. 执行聚合操作的线程为连续线程
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
__global__ void reducedVectorAdd2(float *i_data, float *o_data, const uint n) {
    // 计算当前block处理数据块的首地址
    i_data += blockIdx.x * blockDim.x;
    
    uint tid = threadIdx.x;
    // stride代表含义：
    // 1. 数据角度：已经聚合完成连续数据块宽度，每个stride的聚合数据点是stride的首元素。例如初始stride为1表示每个元素都聚合了自身，
    //    stride为2时表示聚合了右边一个邻居，聚合点在index为偶数位置，以此类推。
    // 2. 线程角度：当前迭代要聚合相邻两个stride，聚合数据点(data_index)是左边stride的首元素，但是执行聚合的线程是一个block内的连续线程，
    //    即0号线程聚合0到0和1号stride，1号线程聚合2和3号stride，以此类推
    for (size_t stride=1; stride < blockDim.x; stride<<=1) {
        // 计算聚合数据点在block数据块内偏移
        uint data_index = 2 * stride * tid;
        if (data_index + stride < blockDim.x) {
            i_data[data_index] += i_data[data_index + stride];
        }
        // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证下次迭代之前，上次聚合结果已经写入
        __syncthreads();
    }
    // 最后由0号线程将所在block规约结果写入输出内存
    if (tid == 0) {
        o_data[blockIdx.x] = i_data[0];
    }
}



/**
 * 向量规约求和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 1. 每次聚合固定间隔的数据
 * 2. 执行聚合的线程在block内是连续的
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
__global__ void reducedVectorAdd3(float *i_data, float *o_data, const uint n) {
    // 计算当前block处理数据块的首地址
    i_data += blockIdx.x * blockDim.x;
    
    uint tid = threadIdx.x;
    // stride代表含义：
    // 1. 数据角度：代表待聚合的连续数据块宽度，即stride内的数据点都是聚合点，聚合来自右侧相隔stride的那个数据点
    // 2. 线程角度：stride内的线程执行聚合
    for (size_t stride=blockDim.x/2; stride > 0; stride>>=1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid+stride];
        }
        // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证下次迭代之前，上次聚合结果已经写入
        __syncthreads();
    }
    // 最后由0号线程将所在block规约结果写入输出内存
    if (tid == 0) {
        o_data[blockIdx.x] = i_data[0];
    }
}



/**
 * 向量规约求和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 1. 每次聚合固定间隔的数据
 * 2. 执行聚合的线程在block内是连续的
 * 3. block间数据通过loop unrolling聚合，即一个thread block处理UNROLL_FACTOR个block的数据，UNROLL_Factor支持2,4,8
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
template<uint UNROLL_FACTOR>
__global__ void reducedVectorAddUnrolling1(float *i_data, float *o_data, const uint n) {
    // 计算当前block处理数据块的首地址
    i_data += blockIdx.x * blockDim.x * UNROLL_FACTOR;

    uint tid = threadIdx.x;
    // 先使用循环展开聚合相邻block的数据，block内每个线程聚合相邻block同一个位置的数据(tid)
    float block_merge_val = 0;
    switch (UNROLL_FACTOR)
    {
    case 2:
        block_merge_val = i_data[tid + blockDim.x];
        break;
    case 4:
        block_merge_val = i_data[tid + blockDim.x];
        block_merge_val += i_data[tid + 2 * blockDim.x];
        block_merge_val += i_data[tid + 3 * blockDim.x];
        break;
    case 8:
        block_merge_val = i_data[tid + blockDim.x];
        block_merge_val += i_data[tid + 2 * blockDim.x];
        block_merge_val += i_data[tid + 3 * blockDim.x];
        block_merge_val += i_data[tid + 4 * blockDim.x];
        block_merge_val += i_data[tid + 5 * blockDim.x];
        block_merge_val += i_data[tid + 6 * blockDim.x];
        block_merge_val += i_data[tid + 7 * blockDim.x];
        break;
    default:
        break;
    }
    i_data[tid] += block_merge_val;
    // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证继续前，block内线程都完成了block的unrolling聚合
    __syncthreads();
    
    // stride代表含义：
    // 1. 数据角度：代表待聚合的连续数据块宽度，即stride内的数据点都是聚合点，聚合来自右侧相隔stride的那个数据点
    // 2. 线程角度：stride内的线程执行聚合
    for (size_t stride=blockDim.x/2; stride > 0; stride>>=1) {
        if (tid < stride) {
            i_data[tid] += i_data[tid+stride];
        }
        // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证下次迭代之前，上次聚合结果已经写入
        __syncthreads();
    }
    // 最后由0号线程将所在block规约结果写入输出内存
    if (tid == 0) {
        o_data[blockIdx.x] = i_data[0];
    }
}


/**
 * 向量规约求和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 1. 每次聚合固定间隔的数据
 * 2. 执行聚合的线程在block内是连续的
 * 3. block间数据通过loop unrolling聚合，即一个thread block处理UNROLL_FACTOR个block的数据，UNROLL_Factor支持2,4,8
 * 4. block内规约到最后一个warp时进行loop unrolling
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
template<uint UNROLL_FACTOR>
__global__ void reducedVectorAddUnrolling2(float *i_data, float *o_data, const uint n) {
    // 计算当前block处理数据块的首地址
    i_data += blockIdx.x * blockDim.x * UNROLL_FACTOR;

    uint tid = threadIdx.x;
    // 先使用循环展开聚合相邻block的数据，block内每个线程聚合相邻block同一个位置的数据(tid)
    float block_merge_val = 0;
    switch (UNROLL_FACTOR)
    {
    case 2:
        block_merge_val = i_data[tid + blockDim.x];
        break;
    case 4:
        block_merge_val = i_data[tid + blockDim.x];
        block_merge_val += i_data[tid + 2 * blockDim.x];
        block_merge_val += i_data[tid + 3 * blockDim.x];
        break;
    case 8:
        block_merge_val = i_data[tid + blockDim.x];
        block_merge_val += i_data[tid + 2 * blockDim.x];
        block_merge_val += i_data[tid + 3 * blockDim.x];
        block_merge_val += i_data[tid + 4 * blockDim.x];
        block_merge_val += i_data[tid + 5 * blockDim.x];
        block_merge_val += i_data[tid + 6 * blockDim.x];
        block_merge_val += i_data[tid + 7 * blockDim.x];
        break;
    default:
        break;
    }
    i_data[tid] += block_merge_val;
    // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证继续前，block内线程都完成了block的unrolling聚合
    __syncthreads();
    
    // stride代表含义：
    // 1. 数据角度：代表待聚合的连续数据块宽度，即stride内的数据点都是聚合点，聚合来自右侧相隔stride的那个数据点
    // 2. 线程角度：stride内的线程执行聚合
    for (size_t stride=blockDim.x/2; stride > 32; stride>>=1) {
        // 注意这里stride降低到32时就结束，即留下32个中间聚合结果的数据留给最后一个warp做loop unrolling
        if (tid < stride) {
            i_data[tid] += i_data[tid + stride];
        }
        // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证下次迭代之前，上次聚合结果已经写入
        __syncthreads();
    }
    // 剩余最后一个warp的线程执行聚合，因为SIMT的特性，同一个warp内的线程不需要执行__syncthreads()
    // unrolling loop：就是把前面for循环中的最后5个iteration拆开来
    if (tid < 32) {
        // 这里需要注意：在原文中有说这里是需要加volatile的，因为unroll后的每一步(stride)会读取上一步(stride)的结果，
        // 所以需要保证后一步读取的是有效的数据，所以加上volatile保证每一步聚合结果都能够写入显存，避免被编译器或调度器使用
        // cache优化，但是测试后发现如果加上volatile，性能相比没有unroll版本会略有降低，可能是因为每次访存导致指令stall增加，
        // 去掉volatile测试，则性能相比没有unroll版本就是有提升的，而且也没有发现结果错误的情况，也许指令调度器足够聪明，根据
        // 数据依赖关系能够保证正确性；当然我的测试并不充分，同时不同型号显卡有可能有差别，所以实践中保险起见最好是加上volatile
        VOLATILE float *v_i_data = i_data;
        v_i_data[tid] += v_i_data[tid + 32]; // 对应stride=32
        v_i_data[tid] += v_i_data[tid + 16]; // 对应stride=16
        v_i_data[tid] += v_i_data[tid + 8]; // 对应stride=8
        v_i_data[tid] += v_i_data[tid + 4]; // 对应stride=4
        v_i_data[tid] += v_i_data[tid + 2]; // 对应stride=2
        v_i_data[tid] += v_i_data[tid + 1]; // 对应stride=1
    }
    // 最后由0号线程将所在block规约结果写入输出内存
    if (tid == 0) {
        o_data[blockIdx.x] = i_data[0];
    }
}


/**
 * 规约向量和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 1. 每次聚合固定间隔的数据
 * 2. 执行聚合的线程在block内是连续的
 * 3. block间数据通过loop unrolling聚合，即一个thread block处理UNROLL_FACTOR个block的数据，UNROLL_Factor支持2,4,8
 * 4. block内全部loop unrolling
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
template<uint UNROLL_FACTOR, uint BLOCK_SIZE=512>
__global__ void reducedVectorAddUnrolling3(float *i_data, float *o_data, const uint n) {
    // 计算当前block处理数据块的首地址
    i_data += blockIdx.x * blockDim.x * UNROLL_FACTOR;

    uint tid = threadIdx.x;
    // 先使用循环展开聚合相邻block的数据，block内每个线程聚合相邻block同一个位置的数据(tid)
    float block_merge_val = 0;
    switch (UNROLL_FACTOR)
    {
    case 2:
        block_merge_val = i_data[tid + blockDim.x];
        break;
    case 4:
        block_merge_val = i_data[tid + blockDim.x];
        block_merge_val += i_data[tid + 2 * blockDim.x];
        block_merge_val += i_data[tid + 3 * blockDim.x];
        break;
    case 8:
        block_merge_val = i_data[tid + blockDim.x];
        block_merge_val += i_data[tid + 2 * blockDim.x];
        block_merge_val += i_data[tid + 3 * blockDim.x];
        block_merge_val += i_data[tid + 4 * blockDim.x];
        block_merge_val += i_data[tid + 5 * blockDim.x];
        block_merge_val += i_data[tid + 6 * blockDim.x];
        block_merge_val += i_data[tid + 7 * blockDim.x];
        break;
    default:
        break;
    }
    i_data[tid] += block_merge_val;
    // 因为聚合结果是in-place写入输入向量，所以需要对block内线程进行一次同步，保证继续前，block内线程都完成了block的unrolling聚合
    __syncthreads();
    
    // block内线程全部unrolling
    if (BLOCK_SIZE==1024 && tid < 512) {
        // 对应stride=512
        i_data[tid] += i_data[tid + 512];
        __syncthreads();
    }
    if (BLOCK_SIZE>=512 && tid < 256) {
        // 对应stride=256
        i_data[tid] += i_data[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE>=256 && tid < 128) {
        // 对应stride=128
        i_data[tid] += i_data[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE>=128 && tid < 64) {
        // 对应stride=64
        i_data[tid] += i_data[tid + 64];
        __syncthreads();
    }
    // 剩余最后一个warp的线程执行聚合，因为SIMT的特性，同一个warp内的线程不需要执行__syncthreads()
    // unrolling loop：就是把前面for循环中的最后5个iteration拆开来
    if (tid < 32) {
        // 这里需要注意：在原文中有说这里是需要加volatile的，因为unroll后的每一步(stride)会读取上一步(stride)的结果，
        // 所以需要保证后一步读取的是有效的数据，所以加上volatile保证每一步聚合结果都能够写入显存，避免被编译器或调度器使用
        // cache优化，但是测试后发现如果加上volatile，性能相比没有unroll版本会略有降低，可能是因为每次访存导致指令stall增加，
        // 去掉volatile测试，则性能相比没有unroll版本就是有提升的，而且也没有发现结果错误的情况，也许指令调度器足够聪明，根据
        // 数据依赖关系能够保证正确性；当然我的测试并不充分，同时不同型号显卡有可能有差别，所以实践中保险起见最好是加上volatile
        VOLATILE float *v_i_data = i_data;
        v_i_data[tid] += v_i_data[tid + 32]; // 对应stride=32
        v_i_data[tid] += v_i_data[tid + 16]; // 对应stride=16
        v_i_data[tid] += v_i_data[tid + 8]; // 对应stride=8
        v_i_data[tid] += v_i_data[tid + 4]; // 对应stride=4
        v_i_data[tid] += v_i_data[tid + 2]; // 对应stride=2
        v_i_data[tid] += v_i_data[tid + 1]; // 对应stride=1
    }
    // 最后由0号线程将所在block规约结果写入输出内存
    if (tid == 0) {
        o_data[blockIdx.x] = i_data[0];
    }
}


/**
 * 递归向量规约求和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 采用递归方式，每个block递归启动子grid执行子规约，子grid只包含一个block
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 */
__global__ void reducedVectorAddRecursive1(float *i_data, float *o_data, uint n) {
     // 计算当前block处理数据块的首地址，因为每个block启动的子kernel只包含一个block，
     // 所以只有最顶层的blockIdx.x有不同，其他各级子block的blockIdx.x都为0，
     // 因此各级block的i_data, o_data都是一样的
    float *b_i_data = i_data + blockIdx.x * blockDim.x;
    float *b_o_data = &o_data[blockIdx.x];

    uint tid = threadIdx.x;
    if (blockDim.x == 2 && tid == 0) {
        // 递归终止条件：最低一级block，只剩两个数据，由0号线程聚合后将结果写入输出
        b_o_data[blockIdx.x] = b_i_data[0] + b_i_data[1];
        return;
    }
    uint stride = (blockDim.x >> 1);
    if (stride > 1 && tid < stride) {
        // 执行本级聚合
        b_i_data[tid] += b_i_data[tid + stride];
        // 因为CDP只保证子grid能看到调用父线程（即0号线程）对内存的更改，所以这里是需要__syncthreads()的
        __syncthreads();
        if (tid == 0) {
            // 递归调用，启动子grid
            reducedVectorAddRecursive1<<<1, stride>>>(b_i_data, b_o_data, n);
        }
    }
}

/**
 * 递归向量规约求和，每个block规约一部分数据，block的规约结果写入内存，最后在host端执行所有block规约结果的最终求和：
 * 采用递归方式，由第一个block的第一个线程启动子grid执行子规约，子grid包含相同的block数，只是线程数减半
 * 需要注意的是
 * 1. 每个block处理的数据个数需要是2的幂次数
 * 2. grid及block都是1维的
 * @param i_data 输入向量的首地址，规约结果会in place写入此内存中
 * @param o_data block规约结果内存首地址
 * @param n 向量元素个数
 * @param top_block_size 最顶级grid的block包含线程数，用于各级子grid中计算block的数据偏移 
 */
__global__ void reducedVectorAddRecursive2(float *i_data, float *o_data, uint n, uint top_block_size) {
     // 计算当前block处理数据块的首地址，父子grid包含的block数都是一样的，但是线程数减半，所以这里需要用
     // 顶级grid的block宽度来计算数据偏移
    float *b_i_data = i_data + 2 * blockIdx.x * top_block_size;

    uint tid = threadIdx.x;
    if (blockDim.x == 1) {
        // 递归终止条件：最低一级block，只剩两个数据，由0号线程聚合后将结果写入输出
        o_data[blockIdx.x] = b_i_data[0] + b_i_data[1];
        return;
    }
    // 执行本级聚合
    b_i_data[tid] += b_i_data[tid + blockDim.x];
    // 因为CDP只保证子grid能看到调用父线程对内存的更改，所以这里是需要__syncthreads()的
    __syncthreads();
    if (blockIdx.x == 0 && tid == 0) {
        // 递归调用，启动子grid
        reducedVectorAddRecursive2<<<gridDim.x, blockDim.x/2>>>(i_data, o_data, n, top_block_size);
    }
}