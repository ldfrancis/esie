#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>


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

// Helper macro for cuBLASLt error checking
#define CUBLASLT_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function to read binary file into a vector
std::vector<__half> readBinaryFile(const std::string& filename, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<__half> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(__half));
    return data;
}

void matmulWithCublasLt(cublasLtHandle_t ltHandle,
                        const __half* d_A, const __half* d_B, __half* d_C,
                        int M, int N, int K) {

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    void* workspace = nullptr;
    size_t workspaceSize = 1 << 22; // 4MB

    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

    // Compute in FP32, scaling (alpha/beta) as FP32
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc,
                                            CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F));

    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                        CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                        CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Row‑major layouts: ld = number of columns
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, N));
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Adesc,
                        CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Bdesc,
                        CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Cdesc,
                        CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

    // Heuristic to pick an algorithm
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc,
        Adesc, Bdesc, Cdesc, Cdesc,
        preference, 1, &heuristic, &returned));
    if (returned == 0) {
        std::cerr << "No cuBLASLt algo found" << std::endl;
        exit(EXIT_FAILURE);
    }

    float alpha = 1.0f, beta = 0.0f;

    CUBLASLT_CHECK(cublasLtMatmul(ltHandle,
                                  operationDesc,
                                  &alpha,
                                  d_A, Adesc,
                                  d_B, Bdesc,
                                  &beta,
                                  d_C, Cdesc,
                                  d_C, Cdesc,
                                  &heuristic.algo,   // <-- algorithm pointer
                                  workspace,
                                  workspaceSize,
                                  0));

    // Cleanup
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUDA_CHECK(cudaFree(workspace));
}

int main() {
    // Matrix dimensions
    const int M = 4096; // Rows of A and C
    const int N = 4096; // Columns of B and C
    const int K = 1024; // Columns of A and rows of B

    // Read matrices A, B, and C from binary files
    std::vector<__half> h_A = readBinaryFile("A.bin", M * K);
    std::vector<__half> h_B = readBinaryFile("B.bin", K * N);
    std::vector<__half> target_C = readBinaryFile("C.bin", M * N);
    std::vector<__half> h_C(M * N);

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(__half)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Create cuBLASLt handle
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    // Timing with CUDA events
    const int iters = 500;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        matmulWithCublasLt(ltHandle, d_A, d_B, d_C, M, N, K);
        // Copy result back to host
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
        if (fabsf(__half2float(h_C[i]) - __half2float(target_C[i])) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": "
                      << __half2float(h_C[i]) << " != " << __half2float(target_C[i]) << std::endl;
            // return 1;
        }
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));

    std::cout << "Matrix multiplication completed and result written to C.bin" << std::endl;
    return 0;
}
