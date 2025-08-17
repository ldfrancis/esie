#include <cmath>
#include <limits>
#include "phi.hpp"


PhiForCausalLM::PhiForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps) : 
        hidden_size(hidden_size),
        model(vocab_size, hidden_size, intermediate_size, num_attention_heads, 
        num_key_value_heads, max_position_embeddings, rope_theta, partial_rotary_factor, head_dim, 
        num_hidden_layers, weight, rms_norm_eps), 
        lm_head(hidden_size, vocab_size, nullptr, nullptr) {
    float* weight_itr = weight;
    size_t model_sz = model.get_weight_size();
    weight_itr += model_sz;
    lm_head.set_weight(weight_itr);
}

PhiForCausalLM::~PhiForCausalLM(){
}

void PhiForCausalLM::forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len){
    float* buffer = new float[batch_size*seq_len*this->hidden_size];
    this->model.forward(buffer, input, position_ids, batch_size, seq_len);
    this->lm_head.forward(output, buffer, batch_size, seq_len);
    delete[] buffer;
}

PhiModel::PhiModel(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads,
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
        layers.emplace_back(hidden_size, intermediate_size, num_attention_heads, head_dim, i, num_key_value_heads, nullptr, rms_norm_eps);
    }
    this->set_weight(weight);
}

PhiModel::~PhiModel() {
}

void PhiModel::forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len) {
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

    delete[] embeds;
    delete[] buffer;
    delete[] cos;
    delete[] sin;
}

PhiDecoderLayer::PhiDecoderLayer(int hidden_size, int intermediate_size, int num_attention_heads, int head_dim, int layer_index, int num_key_value_heads, float* weight, float rms_norm_eps):
    hidden_size(hidden_size),
    intermediate_size(intermediate_size), 
    layer_idx(layer_index),
    self_attn_weight(weight), 
    mlp_weight(weight),
    num_attention_heads(num_attention_heads),
    head_dim(head_dim),
    num_key_value_heads(num_key_value_heads), 
    rms_norm_eps(rms_norm_eps),  
    self_attn(hidden_size, num_attention_heads, head_dim, layer_index, num_key_value_heads, nullptr),
    mlp(hidden_size, intermediate_size, nullptr),
    input_layernorm(nullptr, rms_norm_eps, hidden_size),
    post_attention_layernorm(nullptr, rms_norm_eps, hidden_size) {
    if (weight) this->set_weight(weight);
}

PhiDecoderLayer::~PhiDecoderLayer() {
}

void PhiDecoderLayer::forward(float* output, const float* input, const float* cos, const float* sin, 
    int batch_size, int seq_len) {
    float* temp_output = new float[batch_size * seq_len * hidden_size];
    float* temp_output1 = new float[batch_size * seq_len * hidden_size];

    this->input_layernorm.forward(temp_output, input, batch_size, seq_len);
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

PhiAttention::PhiAttention(int hidden_size, int num_attention_heads, int head_dim, int layer_index, 
    int num_key_value_heads, float* weight)
    : hidden_dim(hidden_size), 
      num_attention_heads(num_attention_heads), 
      hidden_size(hidden_size), 
      layer_index(layer_index), 
      num_key_value_heads(num_key_value_heads),
      head_dim(head_dim),
      scaling(1.0f),
      qkv_proj(hidden_size, (num_attention_heads*head_dim) + (2*num_key_value_heads*head_dim), nullptr, nullptr),
      o_proj(hidden_size, hidden_size, nullptr, nullptr) {
    this->scaling = std::pow(static_cast<float>(this->head_dim), -0.5f);
    this->num_key_value_groups = this->num_attention_heads / this->num_key_value_heads;
}

PhiAttention::~PhiAttention() {
}


void PhiAttention::forward(float* output, const float* input, const float* cos, const float* sin, int batch_size, int seq_len) {
    const int q_dim = this->num_attention_heads * this->head_dim;
    const int kv_dim = this->num_key_value_heads * this->head_dim;
    const int last_dim_length = q_dim + 2 * kv_dim;

    float* q_ = new float[static_cast<size_t>(batch_size) * seq_len * q_dim];
    float* k_ = new float[static_cast<size_t>(batch_size) * seq_len * kv_dim];
    float* qkv = new float[static_cast<size_t>(batch_size) * seq_len * last_dim_length];
    float* buffer = new float[static_cast<size_t>(batch_size) * seq_len * q_dim];
    float* attn_scores_ = new float[static_cast<size_t>(batch_size) * seq_len * seq_len * this->num_attention_heads];

    for (size_t i = 0; i < static_cast<size_t>(batch_size) * seq_len * seq_len * this->num_attention_heads; ++i) attn_scores_[i] = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(batch_size) * seq_len * q_dim; ++i) buffer[i] = 0.0f;
    
    this->qkv_proj.forward(qkv, input, batch_size, seq_len);
    PhiAttention::apply_rotary_pos_embedding(q_, k_, qkv, cos, sin, batch_size, seq_len,
                                             this->num_attention_heads, this->num_key_value_heads, this->head_dim);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int h = 0; h < this->num_attention_heads; ++h) {
                float* query = q_ + (static_cast<size_t>(i) * seq_len + j) * q_dim + h * this->head_dim;
                float* a = attn_scores_ + (static_cast<size_t>(i) * seq_len + j) * (this->num_attention_heads * seq_len) + h * seq_len;
                float max_score = -std::numeric_limits<float>::infinity();

                // Integer KV group index avoids float rounding
                const int kv_group = h / this->num_key_value_groups;

                for (int t = 0; t <= j; ++t) {
                    float* key = k_ + (static_cast<size_t>(i) * seq_len + t) * kv_dim + kv_group * this->head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < this->head_dim; ++d) score += query[d] * key[d];
                    score *= this->scaling;
                    if (score > max_score) max_score = score;
                    a[t] = score;
                }

                float exp_sum = 0.0f;
                for (int t = 0; t <= j; ++t) { a[t] = std::exp(a[t] - max_score); exp_sum += a[t]; }
                const float inv_sum = 1.0f / exp_sum;
                for (int t = 0; t <= j; ++t) a[t] *= inv_sum;

                float* o = buffer + (static_cast<size_t>(i) * seq_len + j) * q_dim + h * this->head_dim;
                for (int t = 0; t <= j; ++t) {
                    float* v = qkv + (static_cast<size_t>(i) * seq_len + t) * last_dim_length
                             + (q_dim + kv_dim) + kv_group * this->head_dim;
                    for (int d = 0; d < this->head_dim; ++d) o[d] += a[t] * v[d];
                }
            }
        }
    }

    this->o_proj.forward(output, buffer, batch_size, seq_len);

    delete[] q_;
    delete[] k_;
    delete[] qkv;
    delete[] attn_scores_;
    delete[] buffer;
}



PhiMLP::PhiMLP(int hidden_size, int intermediate_size, float* /*weight*/)
    : hidden_size(hidden_size), intermediate_size(intermediate_size),
      gate_up_proj(hidden_size, 2*intermediate_size, nullptr, nullptr),
      down_proj(intermediate_size, hidden_size, nullptr, nullptr),
      act_fn() {}


PhiMLP::~PhiMLP() {
}

void PhiMLP::forward(float* output, const float* input, int batch_size, int seq_len) {

    float* up_states = new float[batch_size * seq_len * 2 * intermediate_size];
    float* buffer = new float[batch_size*seq_len*intermediate_size];
    this->gate_up_proj.forward(up_states, input, batch_size, seq_len);

    for(int i=0; i<batch_size; i++){
        for(int j=0; j<seq_len; j++){
            float* gate = up_states + i*seq_len*2*intermediate_size + j*2*intermediate_size;
            float* up = gate + intermediate_size;
            float* b = buffer + i*seq_len*intermediate_size + j*intermediate_size;
            for(int k=0; k<intermediate_size; k++){
                gate[k] = gate[k] / (1.0f + std::exp(-gate[k])); // SiLU activation
                b[k] = gate[k] * up[k]; // Element-wise multiplication
            }

        }
    }

    this->down_proj.forward(output, buffer, batch_size, seq_len);

    // Clean up
    delete[] up_states;
    delete[] buffer;
}

PhiRMSNorm::PhiRMSNorm(float* weight, float variance_epsilon, int hidden_size)
    : weight(weight), variance_epsilon(variance_epsilon), hidden_size(hidden_size) {
}

PhiRMSNorm::~PhiRMSNorm() {
}

void PhiRMSNorm::forward(float* output, const float* input, int batch_size, int seq_len) {
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


PhiRotaryEmbedding::PhiRotaryEmbedding(int hidden_size, int max_position_embeddings, 
    float rope_theta, float partial_rotary_factor, int head_dim):
    max_seq_len_cached(max_position_embeddings), original_max_seq_len(max_position_embeddings),
    rope_theta(rope_theta), partial_rotary_factor(partial_rotary_factor), head_dim(head_dim),
    attention_scaling(1.0f), inv_freqs(nullptr) {
    this->rope_init_fn(rope_theta, partial_rotary_factor, head_dim);
}

PhiRotaryEmbedding::~PhiRotaryEmbedding() {
    delete[] inv_freqs; // Clean up dynamically allocated memory
}

void PhiRotaryEmbedding::rope_init_fn(float rope_theta, float partial_rotary_factor, int head_dim) { 
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

void PhiRotaryEmbedding::forward(float* cos, float* sin, const float* input, const int* position_ids, 
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
