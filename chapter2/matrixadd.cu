#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include "argparse.hpp"
#include "utils.hpp"
#include "timer.hpp"


/**
 * CPU版本矩阵相加：C=A+B
 * @param A A矩阵首地址
 * @param B B矩阵首地址
 * @param C C矩阵首地址
 * @param m 矩阵的行数
 * @param n 矩阵的额列数
 */
void sumMatrixOnHost(const float *A, const float *B, float *C, const uint& m, const uint& n) {
    for (auto j=0; j<m; j++) {
        for (auto i=0; i<n; i++) {
            C[i] = A[i] + B[i];
        }
        // 首地址移动到下一行
        A += n;
        B += n;
        C += n;
    }
}

/**
 * GPU版本矩阵相加：C=A+B
 * @param A A矩阵首地址
 * @param B B矩阵首地址
 * @param C C矩阵首地址
 * @param m 矩阵的行数
 * @param n 矩阵的额列数
 */
__global__ void sumMatrixOnGPU(const float *A, const float *B, float *C, const uint m, const uint n) {
    //------------------------
    // 下面计算线程到数据的映射：
    // 1. 计算线程的全局坐标（i_t, j_t）
    uint i_t = blockIdx.x * blockDim.x + threadIdx.x;
    uint j_t = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. 计算每个线程在x，y方向上分别要处理的数据数：分别将x,y两个方向上的数据数均匀分配到各自方向上的线程数上
    uint x_num_threads = gridDim.x*blockDim.x, y_num_threads = gridDim.y*blockDim.y;
    uint x_per_thread = EVEN_DIVIDE(n, x_num_threads, i_t);
    uint y_per_thread = EVEN_DIVIDE(m, y_num_threads, j_t);

    // 3. 计算每个线程要处理的数据块在x，y方向上的起止坐标：
    uint i_d_min = x_per_thread*i_t, i_d_max = min(i_d_min + x_per_thread, n);
    uint j_d_min = y_per_thread*j_t, j_d_max = min(j_d_min + y_per_thread, m);

    // printf("thread (%d, %d), x_per_thread=%d, y_per_thread=%d, i_d in [%d, %d), j_d in [%d, %d)\n",
    //     i_t, j_t, x_per_thread, y_per_thread, i_d_min, i_d_max, j_d_min, j_d_max);

    // 行优先遍历本线程要处理的数据
    for (auto j_d=j_d_min; j_d<j_d_max; j_d++) {
        for (auto i_d=i_d_min; i_d<i_d_max; i_d++) {
            // 数据逻辑坐标转换为物理坐标
            uint i_p = j_d * n + i_d;
            // printf("thread (%d, %d), data (%d, %d), i_p=%d\n", i_t, j_t, i_d, j_d, i_p);
            // 执行计算
            C[i_p] = A[i_p] + B[i_p];
        }
    } 
}


int main(int argc, char* argv[]) {
    // 解析参数：grid，thread block的形状信息、矩阵的长宽信息
    argparse::ArgumentParser parser("matrixadd");
    std::vector<int> block_dims;
    uint ndim_grid=0, m = 0, n = 0;
    uint round = 1;

    parser.add_argument("--ndim-grid")
        .help("grid的维数，1到2，不指定则与block维度数一致")
        .choices(1, 2)
        .store_into(ndim_grid);

    parser.add_argument("--block")
        .help("thread block形状，空格分割的整数，1到2个，分别代表x,y方向上的大小，几个整数代表几维，默认与block维数一样")
        .required()
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
    
    parser.add_argument("--round")
        .help("测试轮数，跑指定轮数后，取平均值，默认为1")
        .store_into(round);
        
    parser.add_argument("--cpu")
        .help("是否也运行cpu版本，默认不运行")
        .flag();
    
    parser.add_argument("--warmup")
        .help("正式跑之前，是否先warmup，默认不warmup")
        .flag();

    parser.parse_args(argc, argv);

    bool run_cpu = parser.get<bool>("--cpu");
    bool warmup = parser.get<bool>("--warmup");

    assert(std::all_of(block_dims.begin(), block_dims.end(), [](int d){return d>0;}));
    assert(m > 0 && n > 0);
    assert(round > 0);


    printf("--> matrix shape: %dx%d\n", m, n);

    // 为矩阵分配host内存
    uint nelem = m * n;
    size_t nbytes = nelem * sizeof(float);
    float* h_A = (float *)malloc(nbytes);
    float* h_B = (float *)malloc(nbytes);
    float* host_C = (float *)malloc(nbytes);
    float* gpu_C = (float *)malloc(nbytes);
    
    // 随机初始化A，B
    randInitialData(h_A, nelem);
    randInitialData(h_B, nelem);
    // 将C置零
    memset(host_C, 0, nbytes);
    memset(gpu_C, 0, nbytes);

    Timer timer;
    if (run_cpu) {
        if (warmup) {
            // CPU版本预热
            printf("--> start warming up on CPU...\n");
            timer.start();
            sumMatrixOnHost(h_A, h_B, host_C, m, n);
            printf("--> CPU warming up cost %0.6fms\n", timer.elapsed_ms());
        }

        // 执行CPU版本函数
        printf("--> running cpu calculation...\n");
        timer.start();
        sumMatrixOnHost(h_A, h_B, host_C, m, n);
        printf("--> CPU calculation cost %0.6fms\n", timer.elapsed_ms());
    }
    

    // 为矩阵分配GPU内存
    float *d_A, *d_B, *d_C;
    CUDA_CALL_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_B, nbytes));
    CUDA_CALL_CHECK(cudaMalloc(&d_C, nbytes));

    // 将矩阵A，B内容从host拷贝到GPU
    CUDA_CALL_CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));
    CUDA_CALL_CHECK(cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice));

    // 计算设置grid，thread block形状
    dim3 block, grid;
    block.x = block_dims[0];
    if (block_dims.size() > 1) {
        // 2维block
        block.y = block_dims[1];
    }
    switch (ndim_grid)
    {
        case 0:
            // grid维度跟随thread block维度
            for (auto d=0; d<block_dims.size(); d++) {
                if (d == 0) {
                    grid.x = (n + block.x - 1) / block.x;
                }
                if (d == 1) {
                    grid.y = (m + block.y - 1) / block.y;
                }
            }
            break;
        case 1:
            // 1维grid，一行线程总数与数据列数相当
            grid.x = (n + block.x - 1) / block.x;
            break;
        case 2:
            grid.x = (n + block.x - 1) / block.x;
            grid.y = (m + block.y - 1) / block.y;
            break;
        default:
            break;
    }

    printf("--> grid shape: %dx%dx%d, block shape: %dx%dx%d\n", grid.x, grid.y, grid.z, 
        block.x, block.y, block.z);
    if (warmup) {
        // 对CUDA kernel进行预热
        printf("--> start CUDA kernel warming up...\n");
        timer.start();
        sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, m, n);
        CUDA_CALL_CHECK(cudaDeviceSynchronize());
        printf("--> CUDA kernel warming up cost %0.6fms\n", timer.elapsed_ms());
    }

    float total_elapsed_ms = 0;
    for (auto r=0; r<round; r++) {
        // CUDA kernel launch
        printf("--> running round %d GPU calculation...\n", r+1);
        timer.start();
        sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, m, n);
        CUDA_CALL_CHECK(cudaDeviceSynchronize());
        float elpased = timer.elapsed_ms();
        printf("--> round %d GPU calculation cost %0.6fms\n", r+1, elpased);
        total_elapsed_ms += elpased;
    }

    printf("--> GPU calculation avarage cost %0.6fms\n", total_elapsed_ms/round);

    // 将GPU的计算结果拷贝回host
    CUDA_CALL_CHECK(cudaMemcpy(gpu_C, d_C, nbytes, cudaMemcpyDeviceToHost));

    // 校验CPU和GPU计算结果的一致性
    if (run_cpu) {
        checkResult(host_C, gpu_C, nelem);
    }

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