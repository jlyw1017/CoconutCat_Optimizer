#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})

aux_source_directory(. DIR_LIB_SRCS)
add_library (Coconut_Cat_Libs ${DIR_LIB_SRCS})
target_link_libraries(Coconut_Cat_Libs
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY}
        ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY}
        ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY}
        ${CUDA_npps_LIBRARY} ${CUDA_nvcuvenc_LIBRARY}
        ${CUDA_nvcuvid_LIBRARY})
