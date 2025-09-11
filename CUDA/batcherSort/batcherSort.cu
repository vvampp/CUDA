#include <iostream>
#include <algorithm>
#include <cmath>
#define THREADS_PER_BLOCK 256
#define SIZE 8 

__global__ void bitonicSorterHalver(int *list, int j, int k) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        int partner_idx = idx ^ j;

        if (partner_idx > idx) {
                bool sort_ascending = ((idx / k) % 2 == 0);
                if ( (list[idx] > list[partner_idx]) == sort_ascending ) {
                    int temp = list[idx];
                    list[idx] = list[partner_idx];
                    list[partner_idx] = temp;
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

        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int k = 2; k <= n ; k *= 2) 
                for (int j = k / 2 ; j > 0 ; j /= 2) {
                    bitonicSorterHalver<<<blocks, THREADS_PER_BLOCK>>>(device_bitonic_list, j, k);
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
