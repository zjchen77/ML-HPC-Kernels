__global__ void reduce3(float *d_in, float * d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //利用idle线程先做一轮加法
    unsighed int i=blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsighed int tid = threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i+blockDim.x];
    _syncthreads();

    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_out[blockIdx.x] = sdata[tid];
    }
}    //n(logN)