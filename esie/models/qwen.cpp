#include <cmath>
#include <limits>
#include "qwen.hpp"


Qwen3_4BForCausalLM::Qwen3_4BForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps) : 
        hidden_size(hidden_size),
        model(vocab_size, hidden_size, intermediate_size, num_attention_heads, 
        num_key_value_heads, max_position_embeddings, rope_theta, partial_rotary_factor, head_dim, 
        num_hidden_layers, weight, rms_norm_eps), 
        lm_head(hidden_size, vocab_size, weight, nullptr) {

}

Qwen3_4BForCausalLM::~Qwen3_4BForCausalLM(){
    
}

void Qwen3_4BForCausalLM::forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len){
    float* buffer = new float[batch_size*seq_len*this->hidden_size];
    this->model.forward(buffer, input, position_ids, batch_size, seq_len);
    this->lm_head.forward(output, buffer, batch_size, seq_len);
    int h = hidden_size;
    debug_print(output + 2*h-3);
    debug_dash();
    delete[] buffer;
}


Qwen3_8BForCausalLM::Qwen3_8BForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps) : 
        hidden_size(hidden_size),
        model(vocab_size, hidden_size, intermediate_size, num_attention_heads, 
        num_key_value_heads, max_position_embeddings, rope_theta, partial_rotary_factor, head_dim, 
        num_hidden_layers, weight, rms_norm_eps), 
        lm_head(hidden_size, vocab_size, weight, nullptr) {

}

Qwen3_8BForCausalLM::~Qwen3_8BForCausalLM(){
    
}

void Qwen3_8BForCausalLM::forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len){
    float* buffer = new float[batch_size*seq_len*this->hidden_size];
    this->model.forward(buffer, input, position_ids, batch_size, seq_len);
    this->lm_head.forward(output, buffer, batch_size, seq_len);
    int h = hidden_size;
    debug_print(output + 2*h-3);
    debug_dash();
    delete[] buffer;
}


Qwen3_14BForCausalLM::Qwen3_14BForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps) : 
        hidden_size(hidden_size),
        model(vocab_size, hidden_size, intermediate_size, num_attention_heads, 
        num_key_value_heads, max_position_embeddings, rope_theta, partial_rotary_factor, head_dim, 
        num_hidden_layers, weight, rms_norm_eps), 
        lm_head(hidden_size, vocab_size, weight, nullptr) {

}

Qwen3_14BForCausalLM::~Qwen3_14BForCausalLM(){
    
}

void Qwen3_14BForCausalLM::forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len){
    float* buffer = new float[batch_size*seq_len*this->hidden_size];
    this->model.forward(buffer, input, position_ids, batch_size, seq_len);
    this->lm_head.forward(output, buffer, batch_size, seq_len);
    int h = hidden_size;
    debug_print(output + 2*h-3);
    debug_dash();
    delete[] buffer;
}



QwenModel::QwenModel(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads,
     int num_key_value_heads, int max_position_embeddings, float rope_theta, 
     float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
     float rms_norm_eps):
    vocab_size(vocab_size), num_hidden_layers(num_hidden_layers), hidden_size(hidden_size),
    intermediate_size(intermediate_size), head_dim(head_dim), 
    // object initializers
    embed_tokens(vocab_size, hidden_size, weight),
    layers(),
    norm(nullptr, rms_norm_eps, hidden_size),
    rotary_embedding(hidden_size, max_position_embeddings, rope_theta, partial_rotary_factor, head_dim)
{
    for (int i = 0; i < num_hidden_layers; ++i) {
        layers.emplace_back(hidden_size, head_dim, intermediate_size, num_attention_heads, i, num_key_value_heads, nullptr, rms_norm_eps);
    }
    this->set_weight(weight);
}

QwenModel::~QwenModel() {
}

void QwenModel::forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len) {
    float* embeds = new float[batch_size * seq_len * hidden_size];
    float* buffer = new float[batch_size * seq_len * hidden_size];
    this->embed_tokens.forward(embeds, input, batch_size, seq_len);
    float* cos = new float[batch_size*seq_len*this->head_dim];
    float* sin = new float[batch_size*seq_len*this->head_dim];
    this->rotary_embedding.forward(cos, sin, embeds, position_ids, batch_size, seq_len);

    for (auto& layer : layers){
        layer.forward(buffer, embeds, cos, sin, batch_size, seq_len);
        std::swap(embeds, buffer);
    }

    this->norm.forward(output, embeds, batch_size, seq_len);
    int h = hidden_size;
    debug_print(output + 2*h-3);
    debug_dash();
    delete[] embeds;
    delete[] buffer;
    delete[] cos;
    delete[] sin;
}

QwenDecoderLayer::QwenDecoderLayer(int hidden_size, int head_dim, int intermediate_size, int num_attention_heads, int layer_index, int num_key_value_heads, float* weight, float rms_norm_eps):
    hidden_size(hidden_size),
    intermediate_size(intermediate_size), 
    layer_idx(layer_index),
    self_attn_weight(weight), 
    mlp_weight(weight),
    num_attention_heads(num_attention_heads),
    num_key_value_heads(num_key_value_heads), 
    rms_norm_eps(rms_norm_eps),  
    // object initializer
    self_attn(hidden_size, num_attention_heads, layer_index, num_key_value_heads, head_dim, weight, rms_norm_eps),
    mlp(hidden_size, intermediate_size, mlp_weight),
    input_layernorm(weight, rms_norm_eps, hidden_size),
    post_attention_layernorm(weight, rms_norm_eps, hidden_size) {
    if(weight) this->set_weight(weight);
}

QwenDecoderLayer::~QwenDecoderLayer() {
}

void QwenDecoderLayer::forward(float* output, const float* input, const float* cos, const float* sin, int batch_size, int seq_len) {
    
    float* temp_output = new float[batch_size * seq_len * hidden_size];
    float* temp_output1 = new float[batch_size * seq_len * hidden_size];
    this->input_layernorm.forward(temp_output, input, batch_size, seq_len);
    // debug_print(output + 5117);
    this->self_attn.forward(output, temp_output, cos, sin, batch_size, seq_len);

    for(int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        output[i] += input[i]; // Residual connection
    }

    this->post_attention_layernorm.forward(temp_output, output, batch_size, seq_len);
    this->mlp.forward(temp_output1, temp_output, batch_size, seq_len);
    for(int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        output[i] += temp_output1[i]; // Residual connection
    }

    delete[] temp_output;
    delete[] temp_output1;
}


QwenAttention::QwenAttention(int hidden_size, int num_attention_heads, int layer_index, 
    int num_key_value_heads, int head_dim, float* weight, float rms_norm_eps)
    : 
    hidden_dim(hidden_size), 
    num_attention_heads(num_attention_heads), 
    hidden_size(hidden_size), 
    layer_index(layer_index), 
    num_key_value_heads(num_key_value_heads),
    head_dim(head_dim),
    scaling(1.0f),
    // object initializers
    q_proj(hidden_size, num_attention_heads*head_dim, nullptr, nullptr),
    k_proj(hidden_size, num_key_value_heads*head_dim, nullptr, nullptr),
    v_proj(hidden_size, num_key_value_heads*head_dim, nullptr, nullptr),
    o_proj(num_attention_heads*head_dim, hidden_size, nullptr, nullptr),
    q_norm(nullptr, rms_norm_eps, head_dim),
    k_norm(nullptr, rms_norm_eps, head_dim) {

    this->scaling = std::pow(this->head_dim, -0.5f);
    this->num_key_value_groups = num_attention_heads / num_key_value_heads;
}

QwenAttention::~QwenAttention() {
}

void QwenAttention::forward(float* output, const float* input, const float* cos, const float* sin, 
    int batch_size, int seq_len) {
    float* q__ = new float[batch_size * seq_len * num_attention_heads*head_dim];
    float* k__ = new float[batch_size * seq_len * num_key_value_heads*head_dim];
    float* v_ = new float[batch_size * seq_len * num_key_value_heads*head_dim];

    float* q_ = new float[batch_size*seq_len*num_attention_heads*head_dim];
    float* k_ = new float[batch_size*seq_len*num_key_value_heads*head_dim];

    float* attn_scores_ = new float[batch_size * seq_len * seq_len * num_attention_heads];
    for(size_t i=0; i<batch_size*seq_len*seq_len*num_attention_heads; ++i) attn_scores_[i] = 0.0f;

    this->q_proj.forward(q_, input, batch_size, seq_len);
    this->k_proj.forward(k_, input, batch_size, seq_len);
    

    q_norm.forward2(q__, q_, batch_size, seq_len, num_attention_heads);
    
    k_norm.forward2(k__, k_, batch_size, seq_len, num_key_value_heads);
    this->v_proj.forward(v_, input, batch_size, seq_len);
    

    QwenAttention::apply_rotary_pos_embedding(q_, k_, q__, k__, cos, sin, batch_size, seq_len, num_attention_heads, num_key_value_heads, head_dim);
    
    

    float* buffer = new float[batch_size*seq_len*num_attention_heads*head_dim];
    for(size_t i=0; i<batch_size*seq_len*num_attention_heads*head_dim; ++i) {
        buffer[i] = 0.0f; // Initialize buffer to zero
    }
    
    // Compute attention scores and apply softmax
    float kv_group_scale = 1.0f/static_cast<float>(num_key_value_groups);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for(int h=0; h<num_attention_heads; ++h) {
                float* query = q_ + i*seq_len*num_attention_heads*head_dim + j*num_attention_heads*head_dim + h*head_dim;
                float* a = attn_scores_ + i*seq_len*num_attention_heads*seq_len + j*num_attention_heads*seq_len + h*seq_len;
                float max_score = -std::numeric_limits<float>::infinity();

                // obtain dot prod
                for(int k=0; k<j+1; ++k){
                    float* key = k_ + i*seq_len*num_key_value_heads*head_dim + k*num_key_value_heads*head_dim + static_cast<int>(h*kv_group_scale)*head_dim;
                    float score = 0;
                    for(int d=0; d<head_dim; ++d){
                        score += query[d] * key[d];
                    }
                    score *= scaling;
                    if(score > max_score) max_score = score;
                    a[k] = score;

                }

                // exp-score and sum-score
                float exp_sum = 0;
                for(int k=0; k<j+1; ++k){
                    a[k] = std::exp(a[k] - max_score);
                    exp_sum += a[k];
                }

                // Normalize attention scores
                float s = 1/exp_sum;
                for(int k=0; k<j+1; ++k){
                    a[k] *= s;
                }

                // compute attention output
                float* o = buffer + i*seq_len*num_attention_heads*head_dim + j*num_attention_heads*head_dim + h*head_dim;
                for(int k=0; k<j+1; ++k){
                    float* v = v_ + i*seq_len*num_key_value_heads*head_dim + k*num_key_value_heads*head_dim + static_cast<int>(h*kv_group_scale)*head_dim;
                    for(int d=0; d<head_dim; ++d){
                        o[d] += a[k] * v[d];
                    }
                }

            }
        }   
    }

    // Apply the output projection
    this->o_proj.forward(output, buffer, batch_size, seq_len);

    // Clean up
    delete[] q_;
    delete[] k_;
    delete[] v_;
    delete[] q__;
    delete[] k__;
    delete[] attn_scores_;
    delete[] buffer;
    
}


QwenMLP::QwenMLP(int hidden_size, int intermediate_size, float* weight)
    : hidden_size(hidden_size), intermediate_size(intermediate_size),
    // object initializers
      gate_proj(hidden_size, intermediate_size, weight, nullptr),
      up_proj(hidden_size, intermediate_size, weight, nullptr),
      down_proj(intermediate_size, hidden_size, weight, nullptr),
      act_fn() {
    this->set_weight(weight);
}


QwenMLP::~QwenMLP() {
}

void QwenMLP::forward(float* output, const float* input, int batch_size, int seq_len) {

    float* gate_output = new float[batch_size * seq_len * intermediate_size];
    this->gate_proj.forward(gate_output, input, batch_size, seq_len);
    float* up_output = new float[batch_size * seq_len * intermediate_size];
    this->up_proj.forward(up_output, input, batch_size, seq_len);

    this->act_fn.forward(gate_output, gate_output, batch_size, seq_len, intermediate_size);

    // Combine gate and up outputs
    for (int i = 0; i < batch_size * seq_len * intermediate_size; ++i) {
        up_output[i] *= gate_output[i];
    }

    this->down_proj.forward(output, up_output, batch_size, seq_len);

    // Clean up
    delete[] gate_output;
    delete[] up_output;
}

QwenRMSNorm::QwenRMSNorm(float* weight, float variance_epsilon, int hidden_size)
    : weight(weight), variance_epsilon(variance_epsilon), hidden_size(hidden_size) {
}

QwenRMSNorm::~QwenRMSNorm() {
}

void QwenRMSNorm::forward(float* output, const float* input, int batch_size, int seq_len) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float variance = 0.0f;
            for(int k = 0; k < hidden_size; ++k) {
                variance += input[i * seq_len * hidden_size + j * hidden_size + k] * 
                input[i * seq_len * hidden_size + j * hidden_size + k] / hidden_size;
            }
            for(int k = 0; k < hidden_size; ++k){
                output[i * seq_len * hidden_size + j * hidden_size + k] = 
                input[i * seq_len * hidden_size + j * hidden_size + k] *
                    std::sqrt(1.0f / (variance + variance_epsilon)) * weight[k];
            }
        }
    }
    
    
}

void QwenRMSNorm::forward2(float* output, const float* input, int batch_size, int seq_len, int num_heads) {
    int seq = seq_len*num_heads;
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq; ++j) {
            float variance = 0.0f;
            for(int k = 0; k < hidden_size; ++k) {
                variance += input[i * seq * hidden_size + j * hidden_size + k] * 
                input[i * seq * hidden_size + j * hidden_size + k] / hidden_size;
            }
            for(int k = 0; k < hidden_size; ++k){
                output[i * seq * hidden_size + j * hidden_size + k] = 
                input[i * seq * hidden_size + j * hidden_size + k] *
                    std::sqrt(1.0f / (variance + variance_epsilon)) * weight[k];
            }
        }
    }
    
    
}

QwenRotaryEmbedding::QwenRotaryEmbedding(int hidden_size, int max_position_embeddings, 
    float rope_theta, float partial_rotary_factor, int head_dim):
    max_seq_len_cached(max_position_embeddings), original_max_seq_len(max_position_embeddings),
    rope_theta(rope_theta), partial_rotary_factor(partial_rotary_factor), head_dim(head_dim),
    attention_scaling(1.0f), inv_freqs(nullptr) {
    this->rope_init_fn(rope_theta, partial_rotary_factor, head_dim);
}

QwenRotaryEmbedding::~QwenRotaryEmbedding() {
    delete[] inv_freqs; // Clean up dynamically allocated memory
}

void QwenRotaryEmbedding::rope_init_fn(float rope_theta, float partial_rotary_factor, int head_dim) { 
    float base = rope_theta;
    int dim = (int)(head_dim * partial_rotary_factor);
    int half_dim = dim/2;
    delete[] inv_freqs;
    this->inv_freqs = new float[half_dim];

    // Initialize inv_freqs based on the head_dim and rope_theta
    for (int i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2*i) / dim;
        this->inv_freqs[i] = 1.0f / (std::pow(base, exponent));
    }

    this->attention_scaling = 1.0f;
    this->original_inv_freq = this->inv_freqs; // Store the original inv_freqs for potential future use
}

void QwenRotaryEmbedding::forward(float* cos, float* sin, const float* input, const int* position_ids, 
    int batch_size, int seq_len) {

    // Apply rotary embeddings to the input
    const int half_dim = head_dim / 2;
    for(int i=0; i<batch_size; ++i){
        for(int j=0; j<seq_len; ++j){
            const int pos = position_ids ? position_ids[i * seq_len + j] : j;
            for(int k=0; k<half_dim; ++k){
                // Compute the frequency for the current position
                float freq = this->inv_freqs[k] * pos;
                float c = std::cos(freq) * this->attention_scaling;
                float s = std::sin(freq) * this->attention_scaling;
                cos[i * seq_len * head_dim + j * head_dim + k] = c;
                sin[i * seq_len * head_dim + j * head_dim + k] = s;
                // other half
                cos[i * seq_len * head_dim + j * head_dim + (half_dim+k)] = c;
                sin[i * seq_len * head_dim + j * head_dim + (half_dim+k)] = s;
            }
        }
    }
}
