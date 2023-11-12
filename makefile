all: simple interactive

simple: ocl_simple.cpp
	g++ -O2 -I/usr/local/cuda-11/targets/x86_64-linux/include/ -L/usr/local/cuda-11/targets/x86_64-linux/lib/ -o ocl_simple ocl_simple.cpp -lOpenCL
interactive: ocl_interactive.cpp
	g++ -O2 -I/usr/local/cuda-11/targets/x86_64-linux/include/ -L/usr/local/cuda-11/targets/x86_64-linux/lib/ -o ocl_interactive ocl_interactive.cpp -lOpenCL
