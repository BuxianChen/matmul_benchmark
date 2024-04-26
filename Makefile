build: include/mm_impl.hpp src/mm_host.cpp src/mm_utils.cpp src/main.cpp
	g++ -Iinclude/ src/mm_host.cpp src/mm_utils.cpp src/main.cpp -o ./bin/release/main
mm_test: include/mm_impl.hpp src/mm_host.cpp src/mm_utils.cpp src/mm_test.cu
	nvcc -Iinclude/ src/mm_host.cpp src/mm_utils.cpp src/mm_test.cu -o ./bin/release/mm_test