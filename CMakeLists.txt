# cmake最低版本
cmake_minimum_required(VERSION 3.25)
# 设置项目名及语言，包括C++即CUDA代码
project(cuda-prog LANGUAGES CXX CUDA)
# C++标准为17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# 编译过程打印详细信息
set(CMAKE_VERBOSE_MAKEFILE ON)

# 查找cuda
find_package(CUDAToolkit)

# 指定额外的头文件目录
include_directories(common "thirdparty/argparse/include/argparse" "thirdparty/tqdm.cpp/include" "thirdparty/csv-parser/include")

add_compile_options(-g)
# 设置编译目标arch（compute capability），我的显卡（RTX 2060）支持到7.5
set(CMAKE_CUDA_ARCHITECTURES 75)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=native>)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
    add_compile_options(-O0)
endif()

option(all "build all chapters" ON)
option(chapter1 "build chapter1")
option(chapter2 "build chapter2")
option(chapter3 "build chapter3")

add_subdirectory(thirdparty/csv-parser)

if(chapter1 OR all)
    add_subdirectory(chapter1)
endif()

if(chapter2 OR all)
    add_subdirectory(chapter2)
endif()

if(chapter3 OR all)
    add_subdirectory(chapter3)
endif()