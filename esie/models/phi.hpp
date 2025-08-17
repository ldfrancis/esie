#pragma once
#include <iostream>
#include <vector>
#include "llama.hpp"


class PhiRotaryEmbedding{
private:
    int max_seq_len_cached;
    int original_max_seq_len;
    float rope_theta;
    float partial_rotary_factor;
    int head_dim;
    float attention_scaling;
    float* inv_freqs;
    float* original_inv_freq;

public:
    PhiRotaryEmbedding(): max_seq_len_cached(0), original_max_seq_len(0), 
        rope_theta(0.0f), partial_rotary_factor(0.0f), head_dim(0), 
        attention_scaling(1.0f), inv_freqs(nullptr), original_inv_freq(nullptr) {}
    PhiRotaryEmbedding(int hidden_size, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim);
    ~PhiRotaryEmbedding();

    void forward(float* cos, float* sin, const float* input, 
        const int* position_ids, int batch_size, int seq_len);
    void rope_init_fn(float rope_theta, float partial_rotary_factor, int head_dim);
};


class PhiRMSNorm{
private:
    float* weight;
    float variance_epsilon;
    int hidden_size;

public:
    PhiRMSNorm(): weight(nullptr), variance_epsilon(1e-6), hidden_size(0) {}
    PhiRMSNorm(float* weight, float variance_epsilon, int hidden_size );
    ~PhiRMSNorm();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        this->weight = new_weight;
    }
    size_t get_weight_size() const{
        size_t size = 0;
        size += hidden_size;
        return size;
    }
};


class PhiMLP{
private:
    int hidden_size;
    int intermediate_size;

    Linear gate_up_proj;
    Linear down_proj;
    SiLU act_fn;

public:
    PhiMLP(): hidden_size(0), intermediate_size(0), gate_up_proj(), down_proj(), act_fn() {}
    PhiMLP(int hidden_size, int intermediate_size, float* weight);
    ~PhiMLP();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        this->gate_up_proj.set_weight(weight_ptr);
        weight_ptr += this->gate_up_proj.get_weight_size();
        this->down_proj.set_weight(weight_ptr);
    }
    size_t get_weight_size() const {
        size_t size = 0;
        size += gate_up_proj.get_weight_size();
        size += down_proj.get_weight_size();
        return size;
    }
};


class PhiAttention{
private:
    int hidden_dim;
    int num_attention_heads;
    int head_dim;
    int hidden_size;
    int layer_index;
    int num_key_value_groups;
    int num_key_value_heads;
    float scaling;

    Linear qkv_proj;
    Linear o_proj;

public:
    PhiAttention(): hidden_dim(0), num_attention_heads(0), head_dim(0), hidden_size(0), layer_index(0), num_key_value_groups(0), num_key_value_heads(0), scaling(1.0f) {}
    PhiAttention(int hidden_size, int num_attention_heads, int head_dim, int layer_index, 
        int num_key_value_heads, float* weight);
    ~PhiAttention();

    void forward(float* output, const float* input, const float* cos, const float* sin, 
        int batch_size, int seq_len);

    void set_weight(float* new_weight) {
        // Weight order per comments:
        // 1) self_attn.o_proj.weight
        // 2) self_attn.qkv_proj.weight
        float* weight_ptr = new_weight;
        this->o_proj.set_weight(weight_ptr);
        weight_ptr += o_proj.get_weight_size();
        this->qkv_proj.set_weight(weight_ptr);
    }

    size_t get_weight_size() const {
        return o_proj.get_weight_size() + qkv_proj.get_weight_size();
    }

    static void apply_rotary_pos_embedding(float* q_output, float* k_output, const float* qkv,
         const float* cos, const float* sin, 
        int batch_size, int seq_len, int num_attention_heads, int num_key_value_heads, int head_dim) {
        const int q_dim = num_attention_heads * head_dim;
        const int kv_dim = num_key_value_heads * head_dim;
        const int last_dim_length = q_dim + 2 * kv_dim;

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                const float* cos_seek = cos + (static_cast<size_t>(i) * seq_len + j) * head_dim;
                const float* sin_seek = sin + (static_cast<size_t>(i) * seq_len + j) * head_dim;

                // Q
                for (int h = 0; h < num_attention_heads; ++h) {
                    float* q_out = q_output + (static_cast<size_t>(i) * seq_len + j) * q_dim + h * head_dim;
                    const float* q_in = qkv + (static_cast<size_t>(i) * seq_len + j) * last_dim_length + h * head_dim;

                    const int half = head_dim / 2;
                    for (int d = 0; d < head_dim; ++d) {
                        const int ind_rot = (d + half) % head_dim;
                        const float sgn_rot = (d < half) ? -1.0f : 1.0f;
                        q_out[d] = q_in[d] * cos_seek[d] + sgn_rot * q_in[ind_rot] * sin_seek[d];
                    }
                }

                // K (follows after q_dim)
                for (int h = 0; h < num_key_value_heads; ++h) {
                    float* k_out = k_output + (static_cast<size_t>(i) * seq_len + j) * kv_dim + h * head_dim;
                    const float* k_in = qkv + (static_cast<size_t>(i) * seq_len + j) * last_dim_length + q_dim + h * head_dim;

                    const int half = head_dim / 2;
                    for (int d = 0; d < head_dim; ++d) {
                        const int ind_rot = (d + half) % head_dim;
                        const float sgn_rot = (d < half) ? -1.0f : 1.0f;
                        k_out[d] = k_in[d] * cos_seek[d] + sgn_rot * k_in[ind_rot] * sin_seek[d];
                    }
                }
            }
        }
    }
};


class PhiDecoderLayer{
private:
    int hidden_size;
    int intermediate_size;
    int layer_idx;
    int head_dim;
    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;

    float* self_attn_weight;
    float* mlp_weight;

    PhiAttention self_attn;
    PhiMLP mlp;
    PhiRMSNorm input_layernorm;
    PhiRMSNorm post_attention_layernorm;

public:
    PhiDecoderLayer(): hidden_size(0), intermediate_size(0), layer_idx(0), num_attention_heads(0), head_dim(0), num_key_value_heads(0), rms_norm_eps(0.0f), self_attn_weight(nullptr), mlp_weight(nullptr), self_attn(), mlp(), input_layernorm(), post_attention_layernorm() {}
    PhiDecoderLayer(int hidden_size, int intermediate_size, int num_attention_heads, int head_dim, int layer_index, int num_key_value_heads, float* weight, float rms_norm_eps);
    ~PhiDecoderLayer();


    void forward(float* output, const float* input, const float* cos, const float* sin, int batch_size,
         int seq_len);
         
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        self_attn.set_weight(weight_ptr);
        weight_ptr += self_attn.get_weight_size();
        mlp.set_weight(weight_ptr);
        weight_ptr += mlp.get_weight_size();
        input_layernorm.set_weight(weight_ptr);
        weight_ptr += input_layernorm.get_weight_size();
        post_attention_layernorm.set_weight(weight_ptr);
    }

    size_t get_weight_size() const{
        size_t size = 0;
        // size_t hs = static_cast<size_t>(hidden_size);
        // size_t is = static_cast<size_t>(intermediate_size);
        // size += hs;
        // size += 4ull * hs * hs;
        // size += 3ull * hs * is;
        // size += hs;
        size += self_attn.get_weight_size();
        size += mlp.get_weight_size();
        size += input_layernorm.get_weight_size();
        size += post_attention_layernorm.get_weight_size();
        return size;
    }

};


// Phi Model
class PhiModel{
private:
    int vocab_size;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int head_dim;

    EmbedTokens embed_tokens;
    std::vector<PhiDecoderLayer> layers;
    PhiRMSNorm norm;
    PhiRotaryEmbedding rotary_embedding;

public:
    PhiModel(): vocab_size(0), num_hidden_layers(0), hidden_size(0), intermediate_size(0), head_dim(0), embed_tokens(), layers(), norm(), rotary_embedding() {}
    PhiModel(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~PhiModel();

    void forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_itr = new_weight;
        this->embed_tokens.set_weight(weight_itr);
        weight_itr += static_cast<size_t>(vocab_size) * static_cast<size_t>(hidden_size);
        for(auto& layer : layers) {
            layer.set_weight(weight_itr);
            weight_itr += layer.get_weight_size();
            // weight_itr += hidden_size; // input layer norm
            // weight_itr += 4ull * static_cast<size_t>(hidden_size) * static_cast<size_t>(hidden_size); // attention
            // weight_itr += 3ull * static_cast<size_t>(hidden_size) * static_cast<size_t>(intermediate_size); // MLP
            // weight_itr += hidden_size; // post attention layer norm
        }
        this->norm.set_weight(weight_itr); // Set the weight for the final layer norm
    }

    size_t get_weight_size() const{
        size_t size = 0;
        size += embed_tokens.get_weight_size();
        for (const auto& layer : layers) {
            size += layer.get_weight_size();
        }
        size += norm.get_weight_size();
        return size;
    }
};


// Phi Causal Language Model - Contains the model and lm_head
class PhiForCausalLM{
private:
    int hidden_size;
    Linear lm_head;
    PhiModel model;
public:
    PhiForCausalLM(): hidden_size(0), lm_head(), model() {}
    PhiForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~PhiForCausalLM();

    void set_weight(float* new_weight) {
        this->model.set_weight(new_weight);
        this->lm_head.set_weight(new_weight + model.get_weight_size());
    }

    size_t get_weight_size() const{
        size_t size = 0;
        size += lm_head.get_weight_size();
        size += model.get_weight_size();
        return size;
    }

    void forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len);

};

