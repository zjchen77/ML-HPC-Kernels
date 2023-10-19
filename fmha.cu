#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cmath>
// attention本质就是gemm + softmax + scale + mask   调用cutlass则矩阵乘方面不需要优化了
using ColumnMajor = cutlass::layout::ColumnMajor;

using Gemm = cutlass::gemm::device::Gemm<float,     // 数据类型
                                         ColumnMajor, // A矩阵布局
                                         float,       // 数据类型
                                         ColumnMajor, // B矩阵布局
                                         float,       // 数据类型
                                         ColumnMajor>; // C矩阵布局
// void softmax(float* matrix, int rows, int cols) {
//     for (int i = 0; i < rows; ++i) {
//         float max_val = matrix[i * cols]; // 为了数值稳定性
//         for (int j = 1; j < cols; ++j) {
//             if (matrix[i * cols + j] > max_val)
//                 max_val = matrix[i * cols + j];
//         }

//         float sum = 0.0f;
//         for (int j = 0; j < cols; ++j) {
//             matrix[i * cols + j] = exp(matrix[i * cols + j] - max_val);
//             sum += matrix[i * cols + j];
//         }

//         for (int j = 0; j < cols; ++j) {
//             matrix[i * cols + j] /= sum;
//         }
//     }
// }
// void attention(float* query, float* key, float* value,
//                int sequence_length, int embedding_dim,
//                float* mask, float* output) {
//     int M = sequence_length;
//     int N = sequence_length;
//     int K = embedding_dim;

//     // 计算 Q * K^T
//     float* QK = new float[M * N];
//     Gemm gemm_op;
//     gemm_op({{M, N, K}, {query, M}, {key, K}, {QK, M}, {QK, M}, {1, 0}});

//     // Apply mask
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             QK[i * N + j] *= mask[i * N + j];
//         }
//     }

//     // Scale
//     float scale = 1.0f / sqrt(embedding_dim);
//     for (int i = 0; i < M * N; ++i) {
//         QK[i] *= scale;
//     }

//     // Softmax
//     softmax(QK, M, N);

//     // 计算 softmax(QK) * V
//     gemm_op({{M, N, K}, {QK, M}, {value, K}, {output, M}, {output, M}, {1, 0}});

//     delete[] QK;
// }
#include <cutlass/gemm/device/gemm.h>

using ColumnMajor = cutlass::layout::ColumnMajor;

using Gemm = cutlass::gemm::device::Gemm<float,     // 数据类型
                                         ColumnMajor, // A矩阵布局
                                         float,       // 数据类型
                                         ColumnMajor, // B矩阵布局
                                         float,       // 数据类型
                                         ColumnMajor>; // C矩阵布局

__device__ void mask_and_scale(float* QK, float* mask, int rows, int cols, float scale, int row, int col) {
    if (row < rows && col < cols) {
        if (mask[row * cols + col] == 0) {  
            QK[row * cols + col] = -1e9;   
        }
        QK[row * cols + col] /= scale;     
    }
}

__device__ void rowwise_softmax(float* QK, int rows, int cols, int row) {
    if (row < rows) {
        float max_val = -1e9; 
        float sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            max_val = max(max_val, QK[row * cols + col]);
        }

        for (int col = 0; col < cols; ++col) {
            QK[row * cols + col] = exp(QK[row * cols + col] - max_val);
            sum += QK[row * cols + col];
        }

        for (int col = 0; col < cols; ++col) {
            QK[row * cols + col] /= sum;
        }
    }
}

__global__ void attention_kernel(float* query, float* key, float* value, float* mask, float* output, int sequence_length, int embedding_dim) {
    extern __shared__ float shared_memory[];

    float* QK = shared_memory;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate QK using CUTLASS GEMM
    Gemm gemm_op;
    gemm_op({{sequence_length, sequence_length, embedding_dim}, {query, sequence_length}, {key, embedding_dim}, {QK, sequence_length}, {QK, sequence_length}, {1, 0}});

    __syncthreads();

    float scale = sqrtf(embedding_dim);
    mask_and_scale(QK, mask, sequence_length, sequence_length, scale, row, col);

    __syncthreads();

    rowwise_softmax(QK, sequence_length, sequence_length, row);

    __syncthreads();

    // Multiply (softmax(QK)) * V
    gemm_op({{sequence_length, embedding_dim, sequence_length}, {QK, sequence_length}, {value, sequence_length}, {output, sequence_length}, {output, sequence_length}, {1, 0}});
}

void invoke_attention(float* d_query, float* d_key, float* d_value, float* d_mask, float* d_output, int sequence_length, int embedding_dim) {
    dim3 grid(1);  // for simplicity, adjust as needed
    dim3 block(sequence_length, sequence_length); // assuming it's small enough to fit as block dimensions, adjust if needed

    int shared_memory_size = sequence_length * sequence_length * sizeof(float);

    attention_kernel<<<grid, block, shared_memory_size>>>(d_query, d_key, d_value, d_mask, d_output, sequence_length, embedding_dim);
}


bool invoke_attention(float* h_query, float* h_key, float* h_value, 
                      int sequence_length, int embedding_dim,
                      float* h_mask, float* h_output) {

    float *d_query, *d_key, *d_value, *d_mask, *d_output;
    size_t matrix_size = sequence_length * embedding_dim * sizeof(float);
    size_t mask_size = sequence_length * sequence_length * sizeof(float);

    // 分配 GPU 内存
    cudaMalloc(&d_query, matrix_size);
    cudaMalloc(&d_key, matrix_size);
    cudaMalloc(&d_value, matrix_size);
    cudaMalloc(&d_mask, mask_size);
    cudaMalloc(&d_output, matrix_size);

    // 将数据从 CPU 传输到 GPU
    cudaMemcpy(d_query, h_query, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);

    // 调用在 GPU 上的自注意力算子
    attention(d_query, d_key, d_value, sequence_length, embedding_dim, d_mask, d_output);

    // 将结果从 GPU 传回 CPU
    cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_mask);
    cudaFree(d_output);

    // 对于简化目的，我们假定所有 CUDA 调用都成功。在真实代码中，你应该检查每个 CUDA 调用的返回状态。
    return true;
}
// 假设 h_query, h_key, h_value, h_mask 已经在 CPU 上分配和初始化
float* h_output = new float[sequence_length * embedding_dim];

bool success = invoke_attention(h_query, h_key, h_value, sequence_length, embedding_dim, h_mask, h_output);
if (success) {
    // 使用 h_output
}

delete[] h_output;
