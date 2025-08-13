// Inference Engine for LLMs for CPU/GPU
#include <iostream>
#include <vector>
#include "models/llama.hpp"


int main(int argc, char* argv[]) {
    std::cout << "==================================" << std::endl;
    // weight file and model file 
    // inference engine takes in the weight, model, and input data, then returns the output
    std::cout << "  Hello from ESIE! 🚀" << std::endl;

    LlamaForCausalLM llama_model("path/to/llama/model");
    return 0;
}