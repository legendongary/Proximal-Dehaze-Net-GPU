#include "mex.h"
#include "gpu/mxGPUArray.h"

/* dark channel extraction for single image size of MxNx3 */
void __global__ ex_sm(const float * const A, const float * const B, float * const C, const int M, const int N)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    for(; s<M*N; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M;
        int x = B[0*M*N + n*M + m]-1;
        int y = B[1*M*N + n*M + m]-1;
        int z = B[2*M*N + n*M + m]-1;
        
        C[n*M+m] = A[z*M*N + y*M + x];
    }
}

/* dark channel extraction for multi image size of MxNx3xL */
void __global__ ex_mm(const float * const A, const float * const B, float * const C, const int M, const int N, const int L)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    int K = 3;
    
    for(; s<M*N*L; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M % N;
        int l = (s-n*M-m) / (M*N);
        int x = B[l*M*N*K + 0*M*N + n*M + m]-1;
        int y = B[l*M*N*K + 1*M*N + n*M + m]-1;
        int z = B[l*M*N*K + 2*M*N + n*M + m]-1;
        
        C[l*M*N + n*M + m] = A[l*M*N*K + z*M*N + y*M + x];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    mxGPUArray const *A;
    mxGPUArray const *B;
    mxGPUArray       *C;
    float const *d_A;
    float const *d_B;
    float       *d_C;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    int const threadsPerBlock = 512;
    int blocksPerGrid;
    
    mxInitGPU();
    
    if((nrhs!=2) || !(mxIsGPUArray(prhs[0])))
        mexErrMsgIdAndTxt(errId, errMsg);
    
    A = mxGPUCreateFromMxArray(prhs[0]);
    B = mxGPUCreateFromMxArray(prhs[1]);
    
    if(mxGPUGetClassID(A)!=mxSINGLE_CLASS || mxGPUGetClassID(B)!=mxSINGLE_CLASS)
        mexErrMsgIdAndTxt(errId, errMsg);
    
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    d_B = (float const *)(mxGPUGetDataReadOnly(B));
    
    const mwSize ndims_A = mxGPUGetNumberOfDimensions(A);
    
    if(ndims_A==3)
    {
        const mwSize *dims_A = mxGPUGetDimensions(A);
        const size_t M = dims_A[0];
        const size_t N = dims_A[1];
        const mwSize dims_C[] = {M, N};
        C = mxGPUCreateGPUArray(2, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_C = (float *)(mxGPUGetData(C));
        blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
        ex_sm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
    }
    else if(ndims_A==4)
    {
        const mwSize *dims_A = mxGPUGetDimensions(A);
        const size_t M = dims_A[0];
        const size_t N = dims_A[1];
        const size_t L = dims_A[3];
        const mwSize dims_C[] = {M, N, 1, L};
        C = mxGPUCreateGPUArray(4, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_C = (float *)(mxGPUGetData(C));
        blocksPerGrid = (M*N*L + threadsPerBlock - 1) / threadsPerBlock;
        ex_mm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, L);
    }
    else
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
    
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
}