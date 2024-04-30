build: include/mm_impl.cuh src/mm_host.cpp src/mm_v0.cu src/mm_v1.cu src/mm_utils.cpp src/cuda_helper.cu src/main.cu
	nvcc -Iinclude/ src/mm_host.cpp src/mm_utils.cpp src/mm_v0.cu src/mm_v1.cu src/cuda_helper.cu src/main.cu -o ./bin/release/main
