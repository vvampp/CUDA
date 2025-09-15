#include <iostream>
#include <algorithm>
#include <cmath>
#include "utils.h"
#define THREADS_PER_BLOCK 512 

__global__ void bitonicSorterHalver(float *list, int j, int k) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        int dir = ((idx & k) == 0) ? 1 : 0;

        int p_idx = idx ^ j;

        if (p_idx > idx) 
                if (dir == 1) {
                        if (list[idx] > list[p_idx]) {
                                float temp = list[idx];
                                list[idx] = list[p_idx];
                                list[p_idx] = temp;
                        }
                }else{
                        if (list[idx] < list[p_idx]) {
                                float temp = list[idx];
                                list[idx] = list[p_idx];
                                list[p_idx] = temp;
                        }
                }
}

int main(int argc, char* argv[]){
        if(argc < 2){
                std::cout << "Use: " << argv[0] << " <input_file_name.txt>" << std::endl;
                return -1;
        }

        long n = 0;
        float *array = txt_to_array(argv[1], n);

        if(array == NULL){
                std::cout << "Couldn't read numbers from " << argv[1] << std::endl;
                return -1;
        }

        float * device_array; 
        cudaMalloc((void**)&device_array,sizeof(float)*n);
        cudaMemcpy(device_array,array,sizeof(float)*n,cudaMemcpyHostToDevice);

        int threads = std::min((int)n,THREADS_PER_BLOCK);
        int blocks = (n + threads - 1) / threads;

        for (int k = 2; k <= n; k *= 2)
                for (int j = k / 2 ; j > 0 ; j /= 2) {
                    bitonicSorterHalver<<<blocks, threads>>>(device_array, j, k);
                    cudaDeviceSynchronize();
                }

        cudaMemcpy(array,device_array,sizeof(float)*n,cudaMemcpyDeviceToHost);
        array_to_txt("sorted_numbers.txt",array,n);
        cudaFree(device_array);

        return 0;
}

