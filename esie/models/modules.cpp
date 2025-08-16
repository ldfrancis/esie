

class Linear{
private:
    int input_dim;
    int output_dim;
    float* weight;
    float* bias;
public:
    Linear(): input_dim(0), output_dim(0), weight(nullptr), bias(nullptr) {}
    Linear(int in_features, int out_features, float* weight, float* bias);
    ~Linear();

    void forward(float* output, const float* input, int batch_size, int seq_len);
    void set_weight(float* new_weight) {
        this->weight = new_weight;
    }
    size_t get_weight_size() const{
        size_t size = 0;
        size += static_cast<size_t>(input_dim) * static_cast<size_t>(output_dim);
        if (bias) {
            size += static_cast<size_t>(output_dim);
        }
        return size;
    }
    size_t get_input_dim(){
        return input_dim;
    }
};

