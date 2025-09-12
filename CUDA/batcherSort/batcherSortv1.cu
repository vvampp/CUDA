#include <iostream>
#include <algorithm>
#include <cmath>
#define THREADS_PER_BLOCK 32 
#define SIZE 32 

__global__ void reverse(int *array, int size){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;        
        while( idx < size / 2){
                int temp = array[idx];
                array[idx] = array[size - 1 - idx];
                array[size - 1 - idx] = temp;
                idx += blockDim.x * gridDim.x;
        }
}

__global__ void bitonicMerge(int *list, int j) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        int p_idx = idx ^ j;

        if (idx < p_idx) {
                if (list[idx] > list[p_idx]) {
                    int temp = list[idx];
                    list[idx] = list[p_idx];
                    list[p_idx] = temp;
                }
        }
}

void bitonicSort(int *d_list, int size){
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

int main(void){
        int n = SIZE;

        int bitonic_list[n] = {3,4,1,6,4,2,6,7,1,32,1,4,5,2,65,7,12,54,65,7,2,6,77,35,32,54,65,7,23,5,0,32};
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";

        int * device_bitonic_list; 
        cudaMalloc((void**)&device_bitonic_list,sizeof(int)*n);
        cudaMemcpy(device_bitonic_list,bitonic_list,sizeof(int)*n,cudaMemcpyHostToDevice);

        bitonicSort(device_bitonic_list,n);

        cudaMemcpy(bitonic_list,device_bitonic_list,sizeof(int)*n,cudaMemcpyDeviceToHost);
        cudaFree(device_bitonic_list);

        printf("\nAfter Bitonic Sorting: \n");
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";
        std::cout << std::endl;

        return 0;
}
