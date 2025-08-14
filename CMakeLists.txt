cmake_minimum_required(VERSION 3.10)

project(ESIE)

set(CMAKE_CXX_STANDARD 14)
# -fsanitize=address -fno-omit-frame-pointer -g to CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")

add_executable(esie main.cpp models/llama.cpp)

target_compile_features(esie PRIVATE cxx_std_14)