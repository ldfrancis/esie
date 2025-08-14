#include <iostream>
#include <vector>




class EmbedTokens{
public:
    EmbedTokens(int vocab_size, int hidden_size, float* weight);
    ~EmbedTokens();

    float* weight;
    int vocab_size;
    int hidden_size;

    void forward(float* output, const int* input_ids, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        this->weight = new_weight;
    }
};


class Linear{
private:
    int input_dim;
    int output_dim;
    float* weight;
    float* bias;
public:
    Linear(int in_features, int out_features, float* weight, float* bias);
    ~Linear();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        this->weight = new_weight;
    }
};


class LlamaRMSNorm{
private:
    float* weight;
    float variance_epsilon;
    int hidden_size;
public:
    LlamaRMSNorm(float* weight, float variance_epsilon, int hidden_size );
    ~LlamaRMSNorm();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        this->weight = new_weight;
    }
};


class LlamaRotaryEmbedding{
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
    LlamaRotaryEmbedding(int hidden_size, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim);
    ~LlamaRotaryEmbedding();

    void forward(float* cos, float* sin, const float* input, 
        const int* position_ids, int batch_size, int seq_len);
    void rope_init_fn(float rope_theta, float partial_rotary_factor, int head_dim);
};


class LlamaAttention{
private:
    int hidden_dim;
    int num_attention_heads;
    int head_dim;
    int hidden_size;
    int layer_index;
    int num_key_value_groups;
    int scaling;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;

public:
    LlamaAttention(int hidden_size, int num_attention_heads, int layer_index, 
        int num_key_value_heads, float* weight);
    ~LlamaAttention();

    void forward(float* output, const float* input, const float* cos, const float* sin, 
        int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        this->q_proj.set_weight(weight_ptr);
        weight_ptr += hidden_size * hidden_size;
        this->k_proj.set_weight(weight_ptr);
        weight_ptr += hidden_size * hidden_size;
        this->v_proj.set_weight(weight_ptr);
        weight_ptr += hidden_size * hidden_size;
        this->o_proj.set_weight(weight_ptr);
    }

    static void apply_rotary_pos_embedding(float* q_output, float* k_output, const float* q,
         const float* k, const float* cos, const float* sin, 
        int batch_size, int seq_len, int num_attention_heads, int head_dim) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                for (int h = 0; h < num_attention_heads; ++h) {
                    float *q_output_seek = q_output + i*seq_len*num_attention_heads*head_dim+
                        j*num_attention_heads*head_dim+h*head_dim;
                    float *k_output_seek = k_output + i*seq_len*num_attention_heads*head_dim+
                        j*num_attention_heads*head_dim+h*head_dim;
                    const float* q_seek = q + i*seq_len*num_attention_heads*head_dim+
                        j*num_attention_heads*head_dim+h*head_dim;
                    const float* k_seek = k + i*seq_len*num_attention_heads*head_dim+
                        j*num_attention_heads*head_dim+h*head_dim;
                    const float* cos_seek = cos + j*head_dim;
                    const float* sin_seek = sin + j*head_dim;
                    int half_dim = head_dim/2;
                    for (int d = 0; d < head_dim; ++d) {
                        int ind_rot = (d + half_dim) % head_dim;
                        float sgn_rot = (d < half_dim) ? -1.0f : 1.0f;
                        q_output_seek[d] = q_seek[d] * cos_seek[d] + sgn_rot * q_seek[ind_rot] * sin_seek[d];
                        k_output_seek[d] = k_seek[d] * cos_seek[d] + sgn_rot * k_seek[ind_rot] * sin_seek[d];
                    }
                }
            }
        }
    }
};


class SiLU{
public:
    SiLU();
    ~SiLU();

    void forward(float* output, const float* input, int batch_size, int seq_len);
};


class LlamaMLP{
private:
    int hidden_size;
    int intermediate_size;

    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    SiLU act_fn;

public:
    LlamaMLP(int hidden_size, int intermediate_size, float* weight);
    ~LlamaMLP();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        this->gate_proj.set_weight(weight_ptr);
        weight_ptr += hidden_size * intermediate_size;
        this->up_proj.set_weight(weight_ptr);
        weight_ptr += hidden_size * intermediate_size;
        this->down_proj.set_weight(weight_ptr);
    }
};

void sigmoid(float* output, const float* input, int batch_size, int seq_len);


class LlamaDecoderLayer{
private:
    int hidden_size;
    int layer_idx;
    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;

    float* self_attn_weight;
    float* mlp_weight;

    LlamaAttention self_attn;
    LlamaMLP mlp;
    LlamaRMSNorm input_layernorm;
    LlamaRMSNorm post_attention_layernorm;

public:
    LlamaDecoderLayer(int hidden_size, int intermediate_size, int num_attention_heads, int layer_index, int num_key_value_heads, float* weight, float rms_norm_eps);
    ~LlamaDecoderLayer();


    void forward(float* output, const float* input, const float* cos, const float* sin, int batch_size,
         int seq_len);
    void set_weight(float* new_weight) {
        float* weight_ptr = new_weight;
        this->input_layernorm.set_weight(weight_ptr);
        weight_ptr += this->hidden_size;
        this->self_attn.set_weight(weight_ptr);
        weight_ptr += 4 * hidden_size * hidden_size;
        this->mlp.set_weight(weight_ptr);
        weight_ptr += 3 * hidden_size * hidden_size;
        this->post_attention_layernorm.set_weight(weight_ptr);
    }

};

class LlamaModel{
private:
    int vocab_size;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int head_dim;

    EmbedTokens embed_tokens;
    std::vector<LlamaDecoderLayer> layers;
    LlamaRMSNorm norm;
    LlamaRotaryEmbedding rotary_embedding;

public:
    LlamaModel(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~LlamaModel();

    void forward(float* output, const int* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        float* weight_itr = new_weight;
        this->embed_tokens.set_weight(weight_itr);
        weight_itr += this->vocab_size * this->hidden_size;
        for(int i=0; i<this->num_hidden_layers; ++i) {
            this->layers[i].set_weight(weight_itr);
            weight_itr = weight_itr + hidden_size; // input layer norm
            weight_itr = weight_itr + 4 * hidden_size * hidden_size; // attention
            weight_itr = weight_itr + 3 * (hidden_size * this->intermediate_size); // MLP
            weight_itr = weight_itr + hidden_size; // post attention layer norm
        }
        this->norm.set_weight(weight_itr); // Set the weight for the final layer norm
    }

    int get_weight_size(){
        int size = 0;
        size += (this->vocab_size*this->hidden_size + 
          this->num_hidden_layers*(this->hidden_size + 4*(this->hidden_size*this->hidden_size) + 
          3*(this->hidden_size * this->intermediate_size) + this->hidden_size) + 
          this->hidden_size
        );
        return size;
    }
};


// Llama model
class LlamaForCausalLM{
private:
    int hidden_size;

    Linear lm_head;
    LlamaModel model;
public:
    LlamaForCausalLM(int vocab_size, int hidden_size, int intermediate_size, int num_attention_heads, 
        int num_key_value_heads, int max_position_embeddings, float rope_theta, 
        float partial_rotary_factor, int head_dim, int num_hidden_layers, float* weight, 
        float rms_norm_eps);
    ~LlamaForCausalLM();

    void set_weight(float* new_weight) {
        this->model.set_weight(new_weight);
        this->lm_head.set_weight(new_weight + model.get_weight_size());
    }

    void forward(float* output, const int* input, int batch_size, int seq_len);

};

