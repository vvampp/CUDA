#include <iostream>
#include <algorithm>
#include <cmath>
#define THREADS_PER_BLOCK 32 
#define SIZE 8 

__global__ void bitonicHalver(int *bitonic_list, int n){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n)
                return;
        if ( bitonic_list[idx] > bitonic_list[idx+n] ) {
                int temp = bitonic_list[idx+n];
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

        int threads = min(sublist_size/2,THREADS_PER_BLOCK);
        int blocks = (sublist_size/2 + threads - 1) / threads; 

        bitonicHalver<<<blocks,threads>>>(sublist_ptr,sublist_size/2);
}


int main(void){
        int n = SIZE;

        int bitonic_list[n] = {1,2,3,6,7,4,2,1};
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";
        int * device_bitonic_list; 
        cudaMalloc((void**)&device_bitonic_list,sizeof(int)*n);
        cudaMemcpy(device_bitonic_list,bitonic_list,sizeof(int)*n,cudaMemcpyHostToDevice);

        for( int sublist_size = n ; sublist_size > 1 ; sublist_size /= 2 ){
                int n_sublists = (int)(n/sublist_size);

                int threads = std::min(n_sublists, THREADS_PER_BLOCK);
                int blocks = (n_sublists + threads - 1) / threads;
                
                bitonicSorter<<<blocks,threads>>>(device_bitonic_list,sublist_size,n_sublists);
                cudaDeviceSynchronize();
        }

        cudaMemcpy(bitonic_list,device_bitonic_list,sizeof(int)*n,cudaMemcpyDeviceToHost);
        cudaFree(device_bitonic_list);

        printf("\nAfter Bitonic Merging: \n");
        for( int i = 0 ; i < n ; ++i )
                std::cout << bitonic_list[i] << " ";
        std::cout << std::endl;

        return 0;
}
