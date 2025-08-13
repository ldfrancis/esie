#include <cmath>
#include "llama.hpp"



LlamaForCausalLM::LlamaForCausalLM(const std::string& model_path){
        std::cout << "Loading Llama model from: " << model_path << std::endl;
        
}


LlamaForCausalLM::~LlamaForCausalLM(){
        std::cout << "Releasing Llama model resources..." << std::endl;
        // TODO
}


LlamaModel::LlamaModel(const std::string& model_path) {
    std::cout << "Loading Llama model from: " << model_path << std::endl;
    this->vocab_size = 32000;
}


LlamaModel::~LlamaModel() {
    std::cout << "Releasing Llama model resources..." << std::endl;
    // TODO
}


EmbedTokens::EmbedTokens(int vocab_size, int hidden_size, float* weight) {
    std::cout << "Loading embedding tokens..." << std::endl;
    this->vocab_size = vocab_size;
    this->hidden_size = hidden_size;
    this->weight = weight;
}


EmbedTokens::~EmbedTokens() {
    std::cout << "Destroying EmbedTokens..." << std::endl;
    // Note: We don't delete weight here since it's managed externally
}


void EmbedTokens::forward(float* output, const int* input_ids, int batch_size, int seq_len) {
    std::cout << "Forward pass through embedding tokens..." << std::endl;
    // select the weights for the input_ids and compute the output
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            int vocab_idx = input_ids[i * seq_len + j];
            for (int k = 0; k < hidden_size; ++k) {
                output[i * seq_len * hidden_size + j * hidden_size + k] = weight[vocab_idx * hidden_size + k];
            }
        }
    }
}


Linear::Linear(int in_features, int out_features, float* weight, float* bias)
    : input_dim(in_features), output_dim(out_features), weight(weight), bias(bias) {
    std::cout << "Initializing Linear layer with input dim: " << in_features << " and output dim: " << out_features << std::endl;
}


Linear::~Linear() {
    std::cout << "Destroying Linear layer..." << std::endl;
    // Note: We don't delete weight and bias here since they're managed externally
}


void Linear::forward(float* output, const float* input, int batch_size, int seq_len) {
    std::cout << "Forward pass through Linear layer..." << std::endl;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < this->output_dim; ++k) {
                float* output1 = output + i * seq_len * this->output_dim + j * this->output_dim;
                const float* input1 = input + i * seq_len * this->input_dim + j * this->input_dim;
                float* weight1 = this->weight + k * this->input_dim;
                output1[k] = 0.0f; // Initialize output to zero
                for (int l = 0; l < this->input_dim; ++l) {
                    output1[k] += input1[l] * weight1[l];
                }
                if (this->bias) {
                    output1[k] += this->bias[k];
                }
            }
        }
    }
}



LlamaAttention::LlamaAttention(int hidden_size, int num_attention_heads, int layer_index, int num_key_value_heads, float* weight)
    : hidden_dim(hidden_size), num_attention_heads(num_attention_heads), hidden_size(hidden_size), layer_index(layer_index), num_key_value_groups(num_key_value_heads),
      q_proj(hidden_size, hidden_size, weight, nullptr),
      k_proj(hidden_size, hidden_size, weight + hidden_size * hidden_size, nullptr),
      v_proj(hidden_size, hidden_size, weight + 2 * hidden_size * hidden_size, nullptr),
      o_proj(hidden_size, hidden_size, weight + 3 * hidden_size * hidden_size, nullptr) {
    std::cout << "Initializing LlamaAttention with hidden size: " << hidden_size << " and number of heads: " << num_attention_heads << std::endl;

    this->scaling = std::pow(this->head_dim, -0.5f);
    this->num_key_value_groups = this->num_attention_heads / num_key_value_heads;
}

LlamaAttention::~LlamaAttention() {
    std::cout << "Destroying LlamaAttention..." << std::endl;
    // Linear objects will be automatically destroyed
}

void LlamaAttention::forward(float* output, const float* input, int batch_size, int seq_len) {
    std::cout << "Forward pass through LlamaAttention...\n To be optimized later" << std::endl;
    // TODO: Optimize the attention mechanism -> flash attention, etc.
    // TODO: Work on the KV Caching
    // Apply the projection layers
    float* q_ = new float[batch_size * seq_len * this->hidden_size];
    float* k_ = new float[batch_size * seq_len * this->hidden_size];
    float* v_ = new float[batch_size * seq_len * this->hidden_size];
    float* attn_scores_ = new float[batch_size * seq_len * seq_len * this->num_attention_heads];

    this->q_proj.forward(q_, input, batch_size, seq_len);
    this->k_proj.forward(k_, input, batch_size, seq_len);
    this->v_proj.forward(v_, input, batch_size, seq_len);

    // Compute attention scores and apply softmax
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float sum_scores = 0.0f;
            for(int k= 0; k < seq_len; ++k){
                for (int h = 0; h < num_attention_heads; ++h) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q_[i * seq_len * hidden_dim + j * hidden_dim + h * head_dim + d] *
                                k_[i * seq_len * hidden_dim + k * hidden_dim + h * head_dim + d];
                    }
                    score *= scaling;
                    score = std::exp(score); 
                    sum_scores += score;
                    attn_scores_[i * seq_len * seq_len * num_attention_heads + j * seq_len * num_attention_heads + k * num_attention_heads + h] = score;
                }
            }
            for (int k = 0; k < seq_len; ++k) {
                for (int h = 0; h < num_attention_heads; ++h) {
                    attn_scores_[i * seq_len * seq_len * num_attention_heads + j * seq_len * num_attention_heads + k * num_attention_heads + h] /= sum_scores;
                }
            }
            // Compute the output for each head by aggregating the values using the scores
            for (int k = 0; k < seq_len; ++k) {
                for (int h = 0; h < num_attention_heads; ++h) {
                    for (int d=0; d < head_dim; ++d){
                        output[i * seq_len * hidden_dim + j * hidden_dim + h * head_dim + d] +=
                            attn_scores_[i * seq_len * seq_len * num_attention_heads + j * seq_len * num_attention_heads + k * num_attention_heads + h] *
                            v_[i * seq_len * hidden_dim + k * hidden_dim + h * head_dim + d];
                    }
                }
            }
        }
    }   

    // Apply the output projection
    this->o_proj.forward(output, output, batch_size, seq_len);

    // Clean up
    delete[] q_;
    delete[] k_;
    delete[] v_;
}


LlamaMLP::LlamaMLP(int hidden_size, int intermediate_size, float* weight)
    : hidden_size(hidden_size), intermediate_size(intermediate_size),
      gate_proj(hidden_size, intermediate_size, weight, nullptr),
      up_proj(hidden_size, intermediate_size, weight + hidden_size * intermediate_size, nullptr),
      down_proj(intermediate_size, hidden_size, weight + 2 * hidden_size * intermediate_size, nullptr),
      act_fn() {
    std::cout << "Initializing LlamaMLP with hidden size: " << hidden_size << " and intermediate size: " << intermediate_size << std::endl;
}

LlamaMLP::~LlamaMLP() {
    std::cout << "Destroying LlamaMLP..." << std::endl;
    // Linear objects will be automatically destroyed
}

void LlamaMLP::forward(float* output, const float* input, int batch_size, int seq_len) {
    std::cout << "Forward pass through LlamaMLP..." << std::endl;

    float* gate_output = new float[batch_size * seq_len * intermediate_size];
    this->gate_proj.forward(gate_output, input, batch_size, seq_len);
    float* up_output = new float[batch_size * seq_len * intermediate_size];
    this->up_proj.forward(up_output, input, batch_size, seq_len);

    this->act_fn.forward(up_output, up_output, batch_size, seq_len);

    // Combine gate and up outputs
    for (int i = 0; i < batch_size * seq_len * intermediate_size; ++i) {
        up_output[i] *= gate_output[i];
    }

    this->down_proj.forward(output, up_output, batch_size, seq_len);

    // Clean up
    delete[] gate_output;
    delete[] up_output;
}


SiLU::SiLU() {
    std::cout << "Initializing SiLU activation function..." << std::endl;
}


SiLU::~SiLU() {
    std::cout << "Destroying SiLU activation function..." << std::endl;
}


void SiLU::forward(float* output, const float* input, int batch_size, int seq_len) {
    std::cout << "Forward pass through SiLU activation function..." << std::endl;

    for (int i = 0; i < batch_size * seq_len; ++i) {
        output[i] = input[i] / (1 + std::exp(-input[i]));
    }
}

LlamaRMSNorm::LlamaRMSNorm(float* weight, float variance_epsilon, int hidden_size)
    : weight(weight), variance_epsilon(variance_epsilon), hidden_size(hidden_size) {
    std::cout << "Initializing LlamaRMSNorm with variance epsilon: " << variance_epsilon << std::endl;
}

LlamaRMSNorm::~LlamaRMSNorm() {
    std::cout << "Destroying LlamaRMSNorm..." << std::endl;
    // Note: We don't delete weight here since it's managed externally
}

void LlamaRMSNorm::forward(float* output, const float* input, int batch_size, int seq_len) {
    std::cout << "Forward pass through LlamaRMSNorm..." << std::endl;

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float variance = 0.0f;
            for(int k = 0; k < hidden_size; ++k) {
                variance += input[i * seq_len * hidden_size + j * hidden_size + k] * input[i * seq_len * hidden_size + j * hidden_size + k] / hidden_size;
            }
            for(int k = 0; k < hidden_size; ++k){
                output[i * seq_len * hidden_size + j * hidden_size + k] = input[i * seq_len * hidden_size + j * hidden_size + k] *
                    std::sqrt(1.0f / (variance + variance_epsilon)) * weight[k];
            }
        }
    }
}


LlamaDecoderLayer::LlamaDecoderLayer(int hidden_size, int intermediate_size, int num_attention_heads, int layer_index, int num_key_value_heads, float* weight, float rms_norm_eps):
    hidden_size(hidden_size), layer_idx(layer_index),
    self_attn_weight(weight), mlp_weight(weight + 4 * hidden_size * hidden_size),
    self_attn(hidden_size, num_attention_heads, layer_index, num_key_value_heads, weight),
    mlp(hidden_size, intermediate_size, mlp_weight),
    input_layernorm(new float[hidden_size], rms_norm_eps, hidden_size),
    post_attention_layernorm(new float[hidden_size], rms_norm_eps, hidden_size) {
    std::cout << "Initializing LlamaDecoderLayer with hidden size: " << hidden_size << ", number of attention heads: " << num_attention_heads << std::endl;
}

LlamaDecoderLayer::~LlamaDecoderLayer() {
    std::cout << "Destroying LlamaDecoderLayer..." << std::endl;
    // Note: We don't delete self_attn_weight and mlp_weight here since they're managed externally
}

void LlamaDecoderLayer::forward(float* output, const float* input, int batch_size, int seq_len) {
    std::cout << "Forward pass through LlamaDecoderLayer..." << std::endl;
    float* temp_output = new float[batch_size * seq_len * hidden_size];

    this->input_layernorm.forward(output, input, batch_size, seq_len);
    this->self_attn.forward(output, output, batch_size, seq_len);

    for(int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        output[i] += input[i]; // Residual connection
    }

    this->post_attention_layernorm.forward(temp_output, output, batch_size, seq_len);
    this->mlp.forward(temp_output, temp_output, batch_size, seq_len);

    for(int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        output[i] += temp_output[i]; // Residual connection
    }

    delete[] temp_output;

}

void sigmoid(float* output, const float* input, int batch_size, int seq_len) {
    for (int i = 0; i < batch_size * seq_len; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}
