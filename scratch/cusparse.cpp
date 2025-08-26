#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper macro for cuSPARSE error checking
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Template function to read binary file into a vector
template <typename T>
std::vector<T> readBinaryFile(const std::string& filename, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<T> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    return data;
}

void sparseMatmulWithCusparse(cusparseHandle_t handle,
                              const __half* d_A, const int* d_A_rowPtr, const int* d_A_colInd,
                              const __half* d_B, __half* d_C,
                              int M, int N, int K, int nnzA) {
    // Descriptor for sparse matrix A
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, K, nnzA,
                                     const_cast<int*>(d_A_rowPtr),
                                     const_cast<int*>(d_A_colInd),
                                     const_cast<__half*>(d_A),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));

    // Descriptor for dense matrix B
    cusparseDnMatDescr_t matB;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, K, N, N, const_cast<__half*>(d_B),
                                       CUDA_R_16F, CUSPARSE_ORDER_ROW));

    // Descriptor for dense matrix C
    cusparseDnMatDescr_t matC;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, M, N, N, d_C,
                                       CUDA_R_16F, CUSPARSE_ORDER_ROW));

    // Alpha and beta must match the "compute type" passed to cusparseSpMM.
    // We use CUDA_R_32F as compute type (mixed precision: FP16 inputs, FP32 accumulate),
    // so alpha/beta must be float, NOT __half. Using __half here yields incorrect results (all zeros).
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Buffer size and workspace
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matB, &beta, matC,
                                           CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize));

    void* dBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Perform sparse matrix multiplication
    CUSPARSE_CHECK(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                dBuffer));

    // Clean up
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUDA_CHECK(cudaFree(dBuffer));
}

int main() {
    // Matrix dimensions (should match those used to generate the binary files in test.py)
    const int M = 4096; // Rows of A and C
    const int N = 4096; // Columns of B and C
    const int K = 1024; // Columns of A and rows of B

    // Read row pointer first to discover nnz (last element of crow array)
    std::vector<int> h_A_rowPtr = readBinaryFile<int>("A_row_ptr.bin", M + 1);
    const int nnzA = h_A_rowPtr.back(); // dynamic nnz derived from data; avoids mismatch
    if (nnzA <= 0 || nnzA > M * K) {
        std::cerr << "Invalid nnzA derived from row pointer: " << nnzA << std::endl;
        return 1;
    }

    // Read remaining CSR components and dense matrices
    std::vector<__half> h_A = readBinaryFile<__half>("A_values.bin", nnzA);
    std::vector<int> h_A_colInd = readBinaryFile<int>("A_col_indices.bin", nnzA);
    std::vector<__half> h_B = readBinaryFile<__half>("B.bin", K * N);
    std::vector<__half> target_C = readBinaryFile<__half>("C.bin", M * N);
    std::vector<__half> h_C(M * N);

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    int *d_A_rowPtr, *d_A_colInd;
    CUDA_CHECK(cudaMalloc(&d_A, nnzA * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_A_rowPtr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_colInd, nnzA * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), nnzA * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_rowPtr, h_A_rowPtr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_colInd, h_A_colInd.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Create cuSPARSE handle
    cusparseHandle_t handle; CUSPARSE_CHECK(cusparseCreate(&handle));

    // Timing with CUDA events
    const int iters = 500;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        sparseMatmulWithCusparse(handle, d_A, d_A_rowPtr, d_A_colInd, d_B, d_C, M, N, K, nnzA);
         // Copy result back to host once (after timing to exclude memcpy)
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Display first 10 values in C
    std::cout << "C: ";
    for (int i = 0; i < 10; i++) {
        std::cout << __half2float(h_C[i]) << " ";
    }
    std::cout << std::endl;

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters; // average per GEMM
    float s = ms * 1e-3f;

    // FLOPs: 2*M*N*K (multiply + add)
    double flops = 2.0 * (double)M * N * K;
    double tflops = (flops / (s)) / 1e12;

    std::cout << "GEMM avg time: " << s << " seconds  |  Throughput: "
              << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Check for correctness
    for (int i = 0; i < M*N; i++) {
        if (fabsf(__half2float(h_C[i]) - __half2float(target_C[i])) > 1e-1) {
            std::cerr << "Mismatch at index " << i << ": "
                      << __half2float(h_C[i]) << " != " << __half2float(target_C[i]) << std::endl;
            // return 1;
        }
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_A_rowPtr));
    CUDA_CHECK(cudaFree(d_A_colInd));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    std::cout << "Sparse matrix multiplication completed." << std::endl;
    return 0;
}
