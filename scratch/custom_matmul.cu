#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)


__global__ void sparse_matmul( __half* C, const __half* A,  const __half* B, int M, int K, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        __half sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            __half a = A[row * K + k];
            if (a) {
                sum += a * B[k * N + col];
            }
        }
        C[row * N + col] = __float2half(sum);
    }
}


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

int main(){
    const int M = 4096;
    const int K = 1024;
    const int N = 4096;

    const int Y = 16;
    const int X = 16;

    __half *dA, *dB, *dC;
    std::vector<__half> h_A = readBinaryFile<__half>("A.bin", M * K);
    std::vector<__half> h_B = readBinaryFile<__half>("B.bin", K * N);
    std::vector<__half> target_C = readBinaryFile<__half>("C.bin", M * N);
    std::vector<__half> h_C(M * N);

    cudaMalloc(&dA, M * K * sizeof(__half));
    cudaMalloc(&dB, K * N * sizeof(__half));
    cudaMalloc(&dC, M * N * sizeof(__half));

    // Copy A and B from host to device
    cudaMemcpy(dA, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);

    // Timing with CUDA events
    const int iters = 500;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        dim3 blockDim(X, Y);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
        sparse_matmul<<<gridDim, blockDim>>>(dC, dA, dB, M, K, N);
        cudaMemcpy(h_C.data(), dC, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
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
        if (fabsf(__half2float(h_C[i]) - __half2float(target_C[i])) > 0.5f) {
            std::cerr << "Mismatch at index " << i << ": "
                      << __half2float(h_C[i]) << " != " << __half2float(target_C[i]) << std::endl;
        }
    }

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}