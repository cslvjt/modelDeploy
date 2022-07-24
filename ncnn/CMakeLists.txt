cmake_minimum_required(VERSION 3.5)
find_package(OpenCV REQUIRED core highgui imgproc)
FIND_PACKAGE(OpenCV)
#ncnn路径
include_directories(/mnt/d/WSL/ncnn/build/install/include/ncnn)
link_directories(/mnt/d/WSL/ncnn/build/install/lib)
include_directories(${OpenCV})
FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

add_executable(main main.cpp)

target_link_libraries(main ncnn ${OpenCV_LIBS})