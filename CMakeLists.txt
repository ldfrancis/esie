cmake_minimum_required(VERSION 3.10)

project(ESIE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")

option(TESTS "Build tests" ON)

include_directories(${CMAKE_SOURCE_DIR})

if(TESTS)
    file(GLOB MODEL_SOURCES "${CMAKE_SOURCE_DIR}/esie/models/*.cpp")
    add_executable(esie-test tests/main.cpp ${MODEL_SOURCES})
    target_compile_features(esie-test PRIVATE cxx_std_14)
else()
    add_executable(esie main.cpp esie/models/llama.cpp esie/models/phi.cpp)
    target_compile_features(esie PRIVATE cxx_std_14)
endif()

