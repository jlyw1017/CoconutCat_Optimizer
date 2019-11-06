cmake_minimum_required(VERSION 3.14)
project(CoconutCat_Optimizer)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Sophus REQUIRED )
include_directories(${Sophus_INCLUDE_DIRS} )

find_package(CUDA)
if (CUDA_FOUND)
    message("CUDA found")
    SET(CUDA_INCLUDE_DIRS "/usr/local/cuda/include" ${CUDA_INCLUDE_DIRS})
    #include(FindCUDA)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
else()
    message("CUDA not found")
endif()

add_executable(CoconutCat_Optimizer main.cpp
        )

target_link_libraries(CoconutCat_Optimizer
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY}
        ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY}
        ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY}
        ${CUDA_npps_LIBRARY} ${CUDA_nvcuvenc_LIBRARY}
        ${CUDA_nvcuvid_LIBRARY})

target_link_libraries(CoconutCat_Optimizer ${OpenCV_LIBS} )
target_link_libraries(CoconutCat_Optimizer ${EIGEN3_LIBRARIES} )
target_link_libraries(CoconutCat_Optimizer ${OpenCV_LIBS} )
