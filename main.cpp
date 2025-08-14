// Inference Engine for LLMs for CPU/GPU
#include <iostream>
#include <vector>
#include "models/llama.hpp"


int main(int argc, char* argv[]) {
    std::cout << "==================================" << std::endl;
    // weight file and model file 
    // inference engine takes in the weight, model, and input data, then returns the output
    std::cout << "  Hello from ESIE! 🚀" << std::endl;

    int vocab_size = 32000; // Example vocabulary size, adjust as needed
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;
    int max_position_embeddings = 4096;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 1.0f;
    int head_dim = 128;
    int num_hidden_layers = 32;
    float rms_norm_eps = 1e-5;

    long total_weight_size = 6738415616;
    float* weight = new float[total_weight_size]; // Placeholder for weight array, should be loaded from a file
    for(long i = 0; i < total_weight_size; ++i) {
        weight[i] = static_cast<float>(i % 100) / 100.0f; // Example initialization, replace with actual weights
    }

    LlamaForCausalLM llama_model(
        vocab_size, hidden_size, intermediate_size, num_attention_heads,
        num_key_value_heads, max_position_embeddings, rope_theta,
        partial_rotary_factor, head_dim, num_hidden_layers, weight, rms_norm_eps
    );

    int input_ids[2][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };
    int position_ids[2][4] = {
        {0, 1, 2, 3},
        {0, 1, 2, 3}
    };
    int batch_size = 2;
    int seq_len = 4;
    float* output = new float[batch_size * seq_len * vocab_size];

    std::cout << "Output from LlamaForCausalLM Before Forward Pass:" << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < 10; ++k) {
                std::cout << output[i * seq_len * vocab_size + j * vocab_size + k] << " ";
            }
            std::cout << std::endl;
        }
    }

    llama_model.forward(output, (const int*)input_ids, (const int*)position_ids, batch_size, seq_len);

    std::cout << "Output from LlamaForCausalLM:" << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < 10; ++k) {
                std::cout << output[i * seq_len * vocab_size + j * vocab_size + k] << " ";
            }
            std::cout << std::endl;
        }
    }

    delete[] weight; // Clean up the weight array
    delete[] output; // Clean up the output array
    return 0;
}