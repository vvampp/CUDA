#include <iostream>
#include <algorithm>
#include <cmath>
#include "utils.h"
#define THREADS_PER_BLOCK 512

__global__ void reverse(float *array, int size){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;        
        while( idx < size / 2){
                float temp = array[idx];
                array[idx] = array[size - 1 - idx];
                array[size - 1 - idx] = temp;
                idx += blockDim.x * gridDim.x;
        }
}

__global__ void bitonicMerge(float *list, int j) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        int p_idx = idx ^ j;

        if (idx < p_idx) {
                if (list[idx] > list[p_idx]) {
                    float temp = list[idx];
                    list[idx] = list[p_idx];
                    list[p_idx] = temp;
                }
        }
}

void bitonicSort(float *d_list, int size){
        if (size <= 1)
                return;
        
        int half_size = size/2;
        int threads = std::min(half_size,THREADS_PER_BLOCK);
        int blocks = (half_size + threads - 1)/threads;

        // Izquierda
        bitonicSort(d_list,half_size);

        // Derecha
        bitonicSort(d_list + half_size,half_size);
        reverse<<<blocks,threads>>>(d_list + half_size,half_size);

        cudaDeviceSynchronize();

        threads = std::min(size,THREADS_PER_BLOCK);
        blocks = (size + threads - 1) / threads;
        for(int j = size / 2 ; j > 0 ; j /= 2){
                bitonicMerge<<<blocks,threads>>>(d_list,j);
                cudaDeviceSynchronize();
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

        bitonicSort(device_array,n);


        cudaMemcpy(array,device_array,sizeof(float)*n,cudaMemcpyDeviceToHost);
        array_to_txt("sorted_numbers.txt",array,n);
        cudaFree(device_array);

        return 0;
}
