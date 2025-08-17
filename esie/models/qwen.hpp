#pragma once
#include <iostream>
#include <vector>
#include "qwen.hpp"
#include "llama.hpp"



class QwenRMSNorm{
private:
    float* weight;
    float variance_epsilon;
    int hidden_size;

public:
    QwenRMSNorm(): weight(nullptr), variance_epsilon(1e-6), hidden_size(0) {}
    QwenRMSNorm(float* weight, float variance_epsilon, int hidden_size );
    ~QwenRMSNorm();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void forward2(float* output, const float* input, int batch_size, int seq_len, int num_heads);
    void set_weight(float* new_weight) {
        this->weight = new_weight;
    }
    size_t get_weight_size() const{
        size_t size = 0;
        size += hidden_size;
        return size;
    }
};


class QwenRotaryEmbedding{
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
    QwenRotaryEmbedding(): max_seq_len_cached(0), original_max_seq_len(0), 
        rope_theta(0.0f), partial_rotary_factor(0.0f), head_dim(0), 
        attention_scaling(1.0f), inv_freqs(nullptr), original_inv_freq(nullptr) {}
    QwenRotaryEmbedding(int hidden_size, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim);
    ~QwenRotaryEmbedding();

    void forward(float* cos, float* sin, const float* input, 
        const int* position_ids, int batch_size, int seq_len);
    void rope_init_fn(float rope_theta, float partial_rotary_factor, int head_dim);
};


class QwenAttention{
private:
    int hidden_dim;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int hidden_size;
    int layer_index;
    int num_key_value_groups;
    float rms_norm_eps;
    float scaling;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;
    QwenRMSNorm q_norm;
    QwenRMSNorm k_norm;

public:
    QwenAttention(): hidden_dim(0), num_attention_heads(0), num_key_value_heads(0), head_dim(0), hidden_size(0), layer_index(0), num_key_value_groups(0), rms_norm_eps(0), scaling(1.0f) {}
    QwenAttention(int hidden_size, int num_attention_heads, int layer_index, 
        int num_key_value_heads, int head_dim, float* weight, float rms_norm_eps);
    ~QwenAttention();

    void forward(float* output, const float* input, const float* cos, const float* sin, 
        int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        this->q_proj.set_weight(weight_ptr);
        weight_ptr += q_proj.get_weight_size();
        this->k_proj.set_weight(weight_ptr);
        weight_ptr += k_proj.get_weight_size();
        this->v_proj.set_weight(weight_ptr);
        weight_ptr += v_proj.get_weight_size();
        this->o_proj.set_weight(weight_ptr);
        weight_ptr += o_proj.get_weight_size();
        this->q_norm.set_weight(weight_ptr);
        weight_ptr += q_norm.get_weight_size();
        this->k_norm.set_weight(weight_ptr);
        weight_ptr += k_norm.get_weight_size();
    }

    size_t get_weight_size() const{
        size_t size = 0;
        size += q_proj.get_weight_size();
        size += k_proj.get_weight_size();
        size += v_proj.get_weight_size();
        size += o_proj.get_weight_size();
        size += q_norm.get_weight_size();
        size += k_norm.get_weight_size();
        return size;
    }

    static void apply_rotary_pos_embedding(float* q_output, float* k_output, const float* q,
         const float* k, const float* cos, const float* sin, 
        int batch_size, int seq_len, int num_attention_heads, int num_key_value_heads, int head_dim) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                const float* cos_seek = cos + j*head_dim;
                const float* sin_seek = sin + j*head_dim;

                for (int h = 0; h < num_attention_heads; ++h) {
                    float *q_output_seek = q_output + i*seq_len*num_attention_heads*head_dim+j*num_attention_heads*head_dim+h*head_dim;
                    const float* q_seek = q + i*seq_len*num_attention_heads*head_dim+j*num_attention_heads*head_dim+h*head_dim;
                    int half_dim = head_dim/2;
                    for (int d = 0; d < head_dim; ++d) {
                        int ind_rot = (d + half_dim) % head_dim;
                        float sgn_rot = (d < half_dim) ? -1.0f : 1.0f;
                        q_output_seek[d] = q_seek[d] * cos_seek[d] + sgn_rot * q_seek[ind_rot] * sin_seek[d];
                        
                    }
                }

                for (int h = 0; h < num_key_value_heads; ++h) {
                    float *k_output_seek = k_output + i*seq_len*num_key_value_heads*head_dim+j*num_key_value_heads*head_dim+h*head_dim;
                    const float* k_seek = k + i*seq_len*num_key_value_heads*head_dim+j*num_key_value_heads*head_dim+h*head_dim;
                    int half_dim = head_dim/2;
                    for (int d = 0; d < head_dim; ++d) {
                        int ind_rot = (d + half_dim) % head_dim;
                        float sgn_rot = (d < half_dim) ? -1.0f : 1.0f;
                        k_output_seek[d] = k_seek[d] * cos_seek[d] + sgn_rot * k_seek[ind_rot] * sin_seek[d];
                    }
                }
            }
        }
    }
};

class QwenMLP{
private:
    int hidden_size;
    int intermediate_size;

    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    SiLU act_fn;

public:
    QwenMLP(): hidden_size(0), intermediate_size(0), gate_proj(), up_proj(), down_proj(), act_fn() {}
    QwenMLP(int hidden_size, int intermediate_size, float* weight);
    ~QwenMLP();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        this->gate_proj.set_weight(weight_ptr);
        weight_ptr += static_cast<size_t>(hidden_size) * static_cast<size_t>(intermediate_size);
        this->up_proj.set_weight(weight_ptr);
        weight_ptr += static_cast<size_t>(hidden_size) * static_cast<size_t>(intermediate_size);
        this->down_proj.set_weight(weight_ptr);
    }
    size_t get_weight_size() const {
        size_t size = 0;
        size += gate_proj.get_weight_size();
        size += up_proj.get_weight_size();
        size += down_proj.get_weight_size();
        return size;
    }
};

void sigmoid(float* output, const float* input, int batch_size, int seq_len);


class QwenDecoderLayer{
private:
    int hidden_size;
    int head_dim;
    int intermediate_size;
    int layer_idx;
    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;

    float* self_attn_weight;
    float* mlp_weight;

    QwenAttention self_attn;
    QwenMLP mlp;
    QwenRMSNorm input_layernorm;
    QwenRMSNorm post_attention_layernorm;

public:
    QwenDecoderLayer(): hidden_size(0), intermediate_size(0), layer_idx(0), num_attention_heads(0), head_dim(0), num_key_value_heads(0), rms_norm_eps(0.0f), self_attn_weight(nullptr), mlp_weight(nullptr), self_attn(), mlp(), input_layernorm(), post_attention_layernorm() {}
    QwenDecoderLayer(int hidden_size, int head_dim, int intermediate_size, int num_attention_heads, int layer_index, int num_key_value_heads, float* weight, float rms_norm_eps);
    ~QwenDecoderLayer();


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
        size += self_attn.get_weight_size();
        size += mlp.get_weight_size();
        size += input_layernorm.get_weight_size();
        size += post_attention_layernorm.get_weight_size();
        return size;
    }

};

class QwenModel{
private:
    int vocab_size;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int head_dim;

    EmbedTokens embed_tokens;
    std::vector<QwenDecoderLayer> layers;
    QwenRMSNorm norm;
    QwenRotaryEmbedding rotary_embedding;

public:
    QwenModel(): vocab_size(0), num_hidden_layers(0), hidden_size(0), intermediate_size(0), head_dim(0), embed_tokens(), layers(), norm(), rotary_embedding() {}
    QwenModel(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~QwenModel();

    void forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_itr = new_weight;
        this->embed_tokens.set_weight(weight_itr);
        weight_itr += static_cast<size_t>(vocab_size) * static_cast<size_t>(hidden_size);
        for(auto& layer : layers) {
            layer.set_weight(weight_itr);
            weight_itr += layer.get_weight_size();
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


// Qwen model
class Qwen3_4BForCausalLM{
protected:
    int hidden_size;

    float* lm_head_weight;
    Linear lm_head;
    QwenModel model;
public:
    Qwen3_4BForCausalLM(): hidden_size(0), lm_head_weight(nullptr), lm_head(), model() {}
    Qwen3_4BForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~Qwen3_4BForCausalLM();

    void set_weight(float* new_weight) {
        this->model.set_weight(new_weight);
        
        this->lm_head.set_weight(new_weight);
    }

    size_t get_weight_size(){
        size_t size = 0;
        size += lm_head.get_weight_size();
        size += model.get_weight_size();
        return size;
    }

    void forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len);
};

class Qwen3_8BForCausalLM{
protected:
    int hidden_size;

    float* lm_head_weight;
    Linear lm_head;
    QwenModel model;
public:
    Qwen3_8BForCausalLM(): hidden_size(0), lm_head_weight(nullptr), lm_head(), model() {}
    Qwen3_8BForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~Qwen3_8BForCausalLM();

    void set_weight(float* new_weight) {
        this->model.set_weight(new_weight);
        this->lm_head.set_weight(new_weight+this->model.get_weight_size());
    }

    size_t get_weight_size(){
        size_t size = 0;
        size += lm_head.get_weight_size();
        size += model.get_weight_size();
        return size;
    }

    void forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len);
};


class Qwen3_14BForCausalLM{
protected:
    int hidden_size;

    float* lm_head_weight;
    Linear lm_head;
    QwenModel model;
public:
    Qwen3_14BForCausalLM(): hidden_size(0), lm_head_weight(nullptr), lm_head(), model() {}
    Qwen3_14BForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~Qwen3_14BForCausalLM();

    void set_weight(float* new_weight) {
        this->model.set_weight(new_weight);
        this->lm_head.set_weight(new_weight+this->model.get_weight_size());
    }

    size_t get_weight_size(){
        size_t size = 0;
        size += lm_head.get_weight_size();
        size += model.get_weight_size();
        return size;
    }

    void forward(float* output, const int* input, const int* position_ids, int batch_size, int seq_len);
};
