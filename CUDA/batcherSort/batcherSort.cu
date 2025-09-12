#include <iostream>
#include <algorithm>
#include <cmath>
#define THREADS_PER_BLOCK 32 
#define SIZE 8 

__global__ void bitonicSorterHalver(int *list, int j) {
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

int main(void){
        int n = SIZE;

        int bitonic_list[n] = {1,2,3,6,7,4,2,1};
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";

        int * device_bitonic_list; 
        cudaMalloc((void**)&device_bitonic_list,sizeof(int)*n);
        cudaMemcpy(device_bitonic_list,bitonic_list,sizeof(int)*n,cudaMemcpyHostToDevice);

        int threads = std::min(n,THREADS_PER_BLOCK);
        int blocks = (n + threads - 1) / threads;

        for (int j = n / 2 ; j > 0 ; j /= 2) {
            bitonicSorterHalver<<<blocks, threads>>>(device_bitonic_list, j);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(bitonic_list,device_bitonic_list,sizeof(int)*n,cudaMemcpyDeviceToHost);
        cudaFree(device_bitonic_list);

        printf("\nAfter Bitonic Sorting: \n");
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";
        std::cout << std::endl;

        return 0;
}
