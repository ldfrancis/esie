#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "esie/models/qwen.hpp"


int test_qwen3_8b() {
    // config ints
    int vocab_size; 
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    int head_dim;
    int num_hidden_layers;

    // config floats
    float rms_norm_eps;
    float rope_theta;
    float partial_rotary_factor;

    std::ifstream file("../weights/Qwen_Qwen3-8B_weights_fp32.bin", std::ios::in|std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }
    std::cout<<"Loading weights from file ../weights/Qwen_Qwen3-8B_weights_fp32.bin" << std::endl;

    size_t bytes_to_read = 256*4;
    char* buffer = new char[bytes_to_read];
    file.read(buffer, bytes_to_read);

    uint32_t* header = reinterpret_cast<uint32_t*>(buffer);
    uint32_t* val;
    vocab_size = (int)(header[2]);
    hidden_size = (int)header[3];
    intermediate_size = (int)header[4];
    num_attention_heads = (int)header[5];
    num_key_value_heads = (int)header[6];
    max_position_embeddings = (int)header[7];
    head_dim = (int)header[8];
    num_hidden_layers = (int)header[9];
    rms_norm_eps = *(reinterpret_cast<float*>(&header[10]));
    rope_theta = *(reinterpret_cast<float*>(&header[11]));
    partial_rotary_factor = *(reinterpret_cast<float*>(&header[12]));

    size_t num_parameters = 0;
    Qwen3_8BForCausalLM l(
        vocab_size, hidden_size, intermediate_size, num_attention_heads,
        num_key_value_heads, max_position_embeddings, rope_theta,
        partial_rotary_factor, head_dim, num_hidden_layers, nullptr, rms_norm_eps
    );
    num_parameters = l.get_weight_size();
    size_t num_bytes = num_parameters * sizeof(float);

    char* buffer2 = new char[num_bytes];
    file.read(buffer2, num_bytes);

    std::cout << "Number of parameters: " << num_parameters << std::endl;
    std::cout << "Assigning model parameters" << std::endl;
    float* weight = reinterpret_cast<float*>(buffer2);
    l.set_weight(weight);
    file.close();

    std::cout << "Forward pass through the model " << std::endl;
    int b=1, s=2;
    int tokens[b][s] = {{4340, 3041}};
    int position_ids[b][s] = {{0, 1}};
    float* logits = new float[b*s*vocab_size];

    // calculate the time for one forward pass
    auto start = std::chrono::high_resolution_clock::now();
    l.forward(logits, (const int*)tokens, (const int*)position_ids, b, s);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time for one forward pass: " << elapsed.count() << " seconds" << std::endl;

    // load logits
    std::cout << "Loading target logits from file ../weights/Qwen_Qwen3-8B_logits.bin" << std::endl;
    std::ifstream file2("../weights/Qwen_Qwen3-8B_logits.bin", std::ios::in|std::ios::binary);
    if(!file2.is_open()){
        std::cerr << "Error opening file!" << std::endl;
        return -1;
    }
    float* target_logits = new float[b*s*vocab_size];
    file2.read(reinterpret_cast<char*>(target_logits), b*s*vocab_size*sizeof(float));
    file2.close();

    // compare logits to target logits
    std::cout << "Checking computed logits and target logits for mismatch" << std::endl;
    bool failed_test = false;
    for(int i=0; i<b*s*vocab_size; i++){
        if(abs(logits[i]-target_logits[i]) > 1e-3){
            std::cout << "Logits mismatch at index " << i << ": "
                      << logits[i] << " != " << target_logits[i] << std::endl;
            failed_test = true;
        }
    }

    if (failed_test) {
        std::cout << "Test Failed" << std::endl;
    } else {
        std::cout << "Test Completed Successfully" << std::endl;
    }

    delete[] buffer;
    delete[] buffer2;
    delete[] logits;
    delete[] target_logits;
    return 0;
}