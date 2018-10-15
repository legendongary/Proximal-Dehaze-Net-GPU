#include "mex.h"
#include "gpu/mxGPUArray.h"

void __global__ up2_ker(const float *const A, float *const B, int const M, int const N)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    int MM = 2*M;
    
    for(; s<M*N; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M;
        
        int b = 2*n*MM + 2*m;
        
        B[b] = A[s];
        B[b+1] = A[s];
        B[b+MM] = A[s];
        B[b+MM+1] = A[s];
    }
}

void __global__ up3_ker(const float *const A, float *const B, int const M, int const N, int const K)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    int MM = 2*M;
    int NN = 2*N;
    
    for(; s<M*N*K; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M % N;
        int k = (s-n*M-m) / (M*N);
        
        int b = k*MM*NN + 2*n*MM + 2*m;
        
        B[b] = A[s];
        B[b+1] = A[s];
        B[b+MM] = A[s];
        B[b+MM+1] = A[s];
    }
}

void __global__ up4_ker(const float *const A, float *const B, int const M, int const N, int const K, int const L)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    int MM = 2*M;
    int NN = 2*N;
    
    for(; s<M*N*K*L; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M % N;
        int k = (s-n*M-m) / (M*N) % K;
        int l = (s-k*M*N-n*M-m) / (M*N*K);
        
        int b = l*MM*NN*K + k*MM*NN + 2*n*MM + 2*m;
        
        B[b] = A[s];
        B[b+1] = A[s];
        B[b+MM] = A[s];
        B[b+MM+1] = A[s];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    mxGPUArray const *A;
    mxGPUArray       *B;
    float const *d_A;
    float       *d_B;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    int const threadsPerBlock = 512;
    int blocksPerGrid;
    
    mxInitGPU();
    
    if((nrhs!=1) || !(mxIsGPUArray(prhs[0])))
        mexErrMsgIdAndTxt(errId, errMsg);
    
    A = mxGPUCreateFromMxArray(prhs[0]);
    
    if(mxGPUGetClassID(A)!=mxSINGLE_CLASS)
        mexErrMsgIdAndTxt(errId, errMsg);
    
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    
    const mwSize ndim_A  = mxGPUGetNumberOfDimensions(A);
    const mwSize *dims_A = mxGPUGetDimensions(A);
    
    size_t M = dims_A[0];
    size_t N = dims_A[1];
    
    if(ndim_A == 2)
    {
        mwSize dims_B[] = {2*M, 2*N};
        B = mxGPUCreateGPUArray(2, dims_B, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_B = (float *)(mxGPUGetData(B));
        blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
        up2_ker<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    }
    else if(ndim_A == 3)
    {
        size_t K = dims_A[2];
        mwSize dims_B[] = {2*M, 2*N, K};
        B = mxGPUCreateGPUArray(3, dims_B, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_B = (float *)(mxGPUGetData(B));
        blocksPerGrid = (M*N*K + threadsPerBlock - 1) / threadsPerBlock;
        up3_ker<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N, K);
    }
    else
    {
        size_t K = dims_A[2];
        size_t L = dims_A[3];
        mwSize dims_B[] = {2*M, 2*N, K, L};
        B = mxGPUCreateGPUArray(4, dims_B, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_B = (float *)(mxGPUGetData(B));
        blocksPerGrid = (M*N*K*L + threadsPerBlock - 1) / threadsPerBlock;
        up4_ker<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N, K, L);
    }
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);
    
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    
}