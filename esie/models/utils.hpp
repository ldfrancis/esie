#ifndef UTILS
#define UTILS

template<typename T>
void debug_print(const T* input){
    for(int i=0; i<3; i++){
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

inline void debug_dash(){
    std::cout << "--------------------" << std::endl;
}
#endif
