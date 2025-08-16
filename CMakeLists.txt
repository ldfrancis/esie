cmake_minimum_required(VERSION 3.10)

project(ESIE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")

if(TESTS)
    include_directories(${CMAKE_SOURCE_DIR})
    add_executable(esie-test tests/main.cpp esie/models/llama.cpp)
else()
add_executable(esie main.cpp esie/models/llama.cpp)
target_compile_features(esie PRIVATE cxx_std_14)
endif()

