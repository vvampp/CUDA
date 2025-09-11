#include <iostream>
#include <algorithm>
#include <cmath>
#define THREADS_PER_BLOCK 256
#define SIZE 8 

__global__ void bitonicHalver(int *bitonic_list, int n){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n)
                return;
        int temp;
        if ( bitonic_list[idx] > bitonic_list[idx+n] ) {
                temp = bitonic_list[idx+n];
                bitonic_list[idx+n] = bitonic_list[idx];
                bitonic_list[idx] = temp;
        }
}

__global__ void bitonicSorter(int *bitonic_list, int sublist_size, int n_sublists){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= n_sublists) 
                return;

        int sublist_index = idx * sublist_size;
        int *sublist_ptr = &bitonic_list[sublist_index];
        int blocks = (sublist_size/2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; 

        bitonicHalver<<<blocks,THREADS_PER_BLOCK>>>(sublist_ptr,sublist_size/2);
}


int main(void){
        int n = SIZE;
        double k = log2(n);

        int bitonic_list[n] = {1,2,3,6,7,4,2,1};
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";

        int * device_bitonic_list; 
        cudaMalloc((void**)&device_bitonic_list,sizeof(int)*n);
        cudaMemcpy(device_bitonic_list,bitonic_list,sizeof(int)*n,cudaMemcpyHostToDevice);

        for( int i = k ; i > 0 ; --i ){
                int sublist_size = pow(2,i);
                int n_sublists = (int)(n/sublist_size);
                int blocks = (n_sublists + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                
                bitonicSorter<<<blocks,THREADS_PER_BLOCK>>>(device_bitonic_list,sublist_size,n_sublists);
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
