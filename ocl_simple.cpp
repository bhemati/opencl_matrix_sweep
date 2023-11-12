#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <CL/cl.h>
#include <vector>
#include <fstream>
#include <string.h>
using namespace std;

#define MAX_SOURCE_SIZE (0x1000)
const int SIZE = 8192;
const int DEPTH = 3;
float A[SIZE][SIZE][3];
float B[SIZE][SIZE][3];
size_t usersize;
int main(int argc, char ** argv)
{
	cout<<"Running sequential part with size "<<SIZE<<" :"<< endl;
	// ############## SEQUENTIAL PART ###################
	//loop to fill in every row
	for(int i=0; i < SIZE; i++)
	{
		//loop to fill in every column
		for(int j=0; j < SIZE; j++)
		{
			A[i][j][0] = (float)i/(float)(j+1.00);
			A[i][j][1] = 1.00;
			A[i][j][2] = (float) j/(float)(i+1.00);
			B[i][j][0] = (float)i/(float)(j+1.00);
			B[i][j][1] = 1.00;
			B[i][j][2] = (float) j/(float)(i+1.00);
		}
	}
	auto t1 = chrono::high_resolution_clock::now();
	//Iteration count
	for (int t=0; t < 24; t++)
	{	
		//each row - beware first row and last row not to be updated therefore from 1...8190
		for(int i=1; i < SIZE -1; i++)
		{
			//each column
			for(int j=0; j < SIZE; j++)
			{
				//only matrix k=1 is updated
				A[i][j][1] += (1 / (sqrt(A[i+1][j][0] + A[i-1][j][2])));
			}
		}
	}
	auto t2 = chrono::high_resolution_clock::now();

	// ############## END OF SEQUENTIAL PART ###################
	std::cout << "Sequential time: " << chrono::duration<double>(t2 - t1).count() << std::endl;
	// std::ofstream ofs("out1.txt", std::ofstream::out);
	// for (int i = 0; i < SIZE; i++)
	// {
	// 	for (int j = 0; j < SIZE; j++)
	// 	{
	// 		ofs << " " << A[i][j][1];
	// 	}
	// 	ofs << endl;
	// }
	// ofs.close();

	
	// ############## PARALLEL PART ###################
	{
		FILE *kernelFile;
		char *kernelSource;
		size_t kernelSize;
		cout<<"Running Parallel part: "<<endl;
		// vector< vector< vector<float> > > A( SIZE , vector< vector<float> > ( SIZE, vector<float> (DEPTH) ) );
		int SZ = SIZE * SIZE * DEPTH * sizeof(float);
		kernelFile = fopen("ocl_simple.cl", "r");

		if (!kernelFile) {

			fprintf(stderr, "No file named ocl_simple.cl was found\n");

			exit(-1);

		}
		kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
		kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
		fclose(kernelFile);

		// Getting platform and device information
		cl_platform_id platformId = NULL;
		cl_device_id deviceID = NULL;
		cl_uint retNumDevices;
		cl_uint retNumPlatforms;
		cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
		ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);
		// Creating context.
		cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);


		// Creating command queue
		cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

		// Memory buffers for each array
		cl_mem bMat = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, SZ, B, &ret);
		cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);	
		ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

		// ############## INIT PART ###################
		auto t1 = chrono::high_resolution_clock::now();

		// Copy lists to memory buffers
		// ret = clEnqueueWriteBuffer(commandQueue, bMat, CL_TRUE, 0, SZ, B, 0, NULL, NULL);
		// // Create kernel
		// cl_kernel kernel = clCreateKernel(program, "initVectors", &ret);
		// // Set arguments for kernel
		// ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMat);	
		// // Execute the kernel
		// size_t globalWorkSize[3] = { SIZE, SIZE, DEPTH};
		// size_t localWorkSize[3] = {16,16,1}; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16 
		// ret = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);	
		// // Read from device back to host.
		// ret = clEnqueueReadBuffer(commandQueue, bMat, CL_TRUE, 0, SZ, B, 0, NULL, NULL);

		// ############## ITER PART ###################
		size_t globalWorkSize2[3] = { SIZE, SIZE, DEPTH};
		size_t localWorkSize2[3] = {1,1,1};
		// ret = clEnqueueWriteBuffer(commandQueue, bMat, CL_TRUE, 0, SZ, B, 0, NULL, NULL);
		cl_kernel kernel = clCreateKernel(program, "iterVectors", &ret);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMat);	
		ret = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, globalWorkSize2, localWorkSize2, 0, NULL, NULL);	
		cout<<ret<<endl;
		ret = clEnqueueReadBuffer(commandQueue, bMat, CL_TRUE, 0, SZ, B, 0, NULL, NULL);

		auto t2 = chrono::high_resolution_clock::now();

		// ############## END OF PARALLEL PART ###################

		std::cout << "GPU (execution only) time: " << chrono::duration<double>(t2 - t1).count() << std::endl;
		// std::ofstream ofs("out3.txt", std::ofstream::out);
		// for (int i = 0; i < SIZE; i++)
		// {
		// 	for (int j = 0; j < SIZE; j++)
		// 	{
		// 		ofs << " " << B[i+ j*SIZE + SIZE*SIZE* 1];
		// 	}
		// 	ofs << endl;
		// }
		// ofs.close();

		// test
		int nq_counter = 0;
		for(int i=0; i < SIZE; i++)
		{
			//loop to fill in every column
			for(int j=0; j < SIZE; j++)
			{
				// using trunc(1000. * lhs) == trunc(1000. * rhs); from stackoverflow
				// because otherwise there would be a lot of false positives since 
				// floating-point arithmetics are weird! 
				if(trunc(1000. * A[i][j][1]) != trunc(1000. *B[i][j][1]))
				{
					nq_counter+=1;
					// cout<<A[i][j][1] <<" "<< B[i][j][1]<< endl;
				}
			}
		}
		cout<< nq_counter<<" out of "<< SIZE * SIZE <<" were NOT exactly equal."<<endl;
		// Clean up, release memory.
		ret = clFlush(commandQueue);
		ret = clFinish(commandQueue);
		ret = clReleaseCommandQueue(commandQueue);
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(bMat);
		ret = clReleaseContext(context);
	}
	return 0;
}