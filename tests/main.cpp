#include "test_llama.hpp"
#include "test_llama3_1_8b.hpp"
#include "test_phi4.hpp"
#include "test_qwen3_4b.hpp"
#include "test_qwen3_8b.hpp"
#include "test_qwen3_14b.hpp"


int main(){
    // std::cout << "Llama2 Test ---" << std::endl;
    // int llama_test = test_llama();

    // std::cout << "Llama3 Test ---" << std::endl;
    // int llama_test3 = test_llama3_1_8b();

    std::cout << "Phi4 Test---" << std::endl;
    int phi4_test = test_phi4();

    // std::cout << "Qwen3 4B Test---" << std::endl;
    // int qwen3_test = test_qwen3_4b();

    // std::cout << "Qwen3 8B Test --" << std::endl;
    // int qwen3_8b_test = test_qwen3_8b();

    // std::cout << "Qwen3 14B Test --" << std::endl;
    // int qwen3_14b_test = test_qwen3_14b();

    return 0;
}