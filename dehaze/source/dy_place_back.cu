#include "mex.h"
#include "gpu/mxGPUArray.h"

/* place back for single image size of MxNx1 */
void __global__ pb_sm(const float *const A, const float *const B, float *const C, float *const D, int const M, int const N, int const W)
{
    int H = (W-1) / 2;
    
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    for(; s<M*N*3; s+=t)
    {
        int m = s % M;
        int n = (s - m) % (M * N) / M;
        int c = (s - n*M - m) / (M * N);
        
        C[s] = 0;
        D[s] = 0;
        float tmpc = 0;
        float tmpd = 0;
        
        for(int p=m-H; p<m+H+1; p++)
        {
            for(int q=n-H; q<n+H+1; q++)
            {
                if(p>-1 && p<M && q>-1 && q<N)
                {
                    if(B[0*M*N+q*M+p]==m+1 && B[1*M*N+q*M+p]==n+1 && B[2*M*N+q*M+p]==c+1)
                    {
                        tmpc += A[q*M+p];
                        tmpd += 1;
                    }
                }
            }
        }
        C[s] = tmpc;
        D[s] = tmpd;
    }
}

/* place back for multi images size of MxNx1xL */
void __global__ pb_mm(const float *const A, const float *const B, float *const C, float *const D, int const M, int const N, int const L, int const W)
{
    int H = (W-1) / 2;
    int K = 3;
    
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    for(; s<M*N*K*L; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M % N;
        int k = (s-n*M-m) / (M*N) % K;
        int l = (s-k*M*N-n*M-m) / (M*N*K);
        
        C[s] = 0;
        D[s] = 0;
        float tmpc = 0;
        float tmpd = 0;
        
        for(int p=m-H; p<m+H+1; p++)
        {
            for(int q=n-H; q<n+H+1; q++)
            {
                if(p>-1 && p<M && q>-1 && q<N)
                {
                    if(B[l*M*N*K+0*M*N+q*M+p]==m+1 && B[l*M*N*K+1*M*N+q*M+p]==n+1 && B[l*M*N*K+2*M*N+q*M+p]==k+1)
                    {
                        tmpc += A[l*M*N+q*M+p];
                        tmpd += 1;
                    }
                }
            }
        }
        C[s] = tmpc;
        D[s] = tmpd;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    mxGPUArray const *A;
    mxGPUArray const *B;
    mxGPUArray       *C;
    mxGPUArray       *D;
    float const *d_A;
    float const *d_B;
    float       *d_C;
    float       *d_D;
    int         W;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    int const threadsPerBlock = 512;
    int blocksPerGrid;
    
    mxInitGPU();
    
    if((nrhs!=3) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])))
        mexErrMsgIdAndTxt(errId, errMsg);
    
    A = mxGPUCreateFromMxArray(prhs[0]);
    B = mxGPUCreateFromMxArray(prhs[1]);
    W = (int)mxGetScalar(prhs[2]);
    d_B = (float const *)(mxGPUGetDataReadOnly(B));
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    
    const mwSize *dims_A = mxGPUGetDimensions(A);
    const mwSize ndims_A = mxGPUGetNumberOfDimensions(A);
    
    if(ndims_A==2)
    {
        size_t M = dims_A[0];
        size_t N = dims_A[1];
        mwSize dims_C[] = {M, N, 3};
        C = mxGPUCreateGPUArray(3, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        D = mxGPUCreateGPUArray(3, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_C = (float *)(mxGPUGetData(C));
        d_D = (float *)(mxGPUGetData(D));
        blocksPerGrid = (M*N*3 + threadsPerBlock - 1) / threadsPerBlock;
        pb_sm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, M, N, W);
    }
    else if(ndims_A==4)
    {
        size_t M = dims_A[0];
        size_t N = dims_A[1];
        size_t L = dims_A[3];
        mwSize dims_C[] = {M, N, 3, L};
        C = mxGPUCreateGPUArray(4, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        D = mxGPUCreateGPUArray(4, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_C = (float *)(mxGPUGetData(C));
        d_D = (float *)(mxGPUGetData(D));
        blocksPerGrid = (M*N*3*L + threadsPerBlock - 1) / threadsPerBlock;
        pb_mm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, M, N, L, W);
    }
    else
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
    plhs[1] = mxGPUCreateMxArrayOnGPU(D);
    
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
    mxGPUDestroyGPUArray(D);
}