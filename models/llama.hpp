#include <iostream>
#include <vector>


// Llama model
class LlamaForCausalLM{
public:
    LlamaForCausalLM(const std::string& model_path);
    ~LlamaForCausalLM();

};


class EmbedTokens{
public:
    EmbedTokens(int vocab_size, int hidden_size, float* weight);
    ~EmbedTokens();

    float* weight;
    int vocab_size;
    int hidden_size;

    void forward(float* output, const int* input_ids, int batch_size, int seq_len);
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
};


class LlamaRotaryEmbedding{
private:
    int max_seq_len_cached;
    int original_max_seq_len;
public:
    LlamaRotaryEmbedding(int hidden_size, float* weight);
    ~LlamaRotaryEmbedding();

    void forward(float* output, const float* input, int batch_size, int seq_len);
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
    LlamaAttention(int hidden_size, int num_attention_heads, int layer_index, int num_key_value_heads, float* weight);
    ~LlamaAttention();

    void forward(float* output, const float* input, int batch_size, int seq_len);
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


    void forward(float* output, const float* input, int batch_size, int seq_len);

};




class LlamaModel{
private:
    int vocab_size;

    EmbedTokens embed_tokens;
    std::vector<LlamaDecoderLayer> layers;
    LlamaRMSNorm norm;

public:
    LlamaModel(int num_hidden_layers);
    ~LlamaModel();

    int vocab_size;
};

