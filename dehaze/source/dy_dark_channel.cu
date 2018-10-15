#include "mex.h"
#include "gpu/mxGPUArray.h"

#define Inf 99999999

/* dark channel for single image size of MxNx3 */
void __global__ dc_sm(const float *const A, float *const B, float *const C, int const M, int const N, int const W)
{
    int H = (W-1) / 2;
    
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    for(; s<M*N; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M;
        
        float tmpc = Inf;
        int tmp1 = 0;
        int tmp2 = 0;
        int tmp3 = 0;
        
        for(int c=0; c<3; c++)
        {
            for(int p=m-H; p<m+H+1; p++)
            {
                for(int q=n-H; q<n+H+1; q++)
                {
                    if(p>-1 && p<M && q>-1 && q<N)
                    {
                        if(A[c*M*N+q*M+p] < tmpc)
                        {
                            tmpc = A[c*M*N+q*M+p];
                            tmp1 = p + 1;
                            tmp2 = q + 1;
                            tmp3 = c + 1;
                        }
                    }
                }
            }
        }
        B[s] = tmpc;
        C[0*M*N + s] = tmp1;
        C[1*M*N + s] = tmp2;
        C[2*M*N + s] = tmp3;
    }
}

/* dark channel for multi images of size MxNx3xL */
void __global__ dc_mm(const float *const A, float *const B, float *const C, int const M, int const N, int const L, int const W)
{
    int H = (W-1) / 2;
    int K = 3;
    
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int t = gridDim.x * blockDim.x;
    
    for(; s<M*N*L; s+=t)
    {
        int m = s % M;
        int n = (s-m) / M % N;
        int l = (s-n*M-m) / (M*N);
        
        float tmpc = Inf;
        int tmp1 = 0;
        int tmp2 = 0;
        int tmp3 = 0;
        
        for(int c=0; c<3; c++)
        {
            for(int p=m-H; p<m+H+1; p++)
            {
                for(int q=n-H; q<n+H+1; q++)
                {
                    if(p>-1 && p<M && q>-1 && q<N)
                    {
                        if(A[l*M*N*K+c*M*N+q*M+p] < tmpc)
                        {
                            tmpc = A[l*M*N*K+c*M*N+q*M+p];
                            tmp1 = p + 1;
                            tmp2 = q + 1;
                            tmp3 = c + 1;
                        }
                    }
                }
            }
        }
        B[s] = tmpc;
        C[l*M*N*K + 0*M*N + n*M + m] = tmp1;
        C[l*M*N*K + 1*M*N + n*M + m] = tmp2;
        C[l*M*N*K + 2*M*N + n*M + m] = tmp3;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    mxGPUArray const *A;
    mxGPUArray       *B;
    mxGPUArray       *C;
    float const *d_A;
    float       *d_B;
    float       *d_C;
    int         W;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    int const threadsPerBlock = 512;
    int blocksPerGrid;
    
    mxInitGPU();
    
    if((nrhs!=2) || !(mxIsGPUArray(prhs[0])))
        mexErrMsgIdAndTxt(errId, errMsg);
    
    A = mxGPUCreateFromMxArray(prhs[0]);
    W = (int)mxGetScalar(prhs[1]);
    
    if(mxGPUGetClassID(A)!=mxSINGLE_CLASS)
        mexErrMsgIdAndTxt(errId, errMsg);
    
    d_A = (float const *)(mxGPUGetDataReadOnly(A));
    
    const mwSize ndims_A = mxGPUGetNumberOfDimensions(A);
    
    if(ndims_A==3)
    {
        const mwSize *dims_A = mxGPUGetDimensions(A);
        const size_t M = dims_A[0];
        const size_t N = dims_A[1];
        const mwSize dims_B[] = {M, N};
        const mwSize dims_C[] = {M, N, 3};
        B = mxGPUCreateGPUArray(2, dims_B, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        C = mxGPUCreateGPUArray(3, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_B = (float *)(mxGPUGetData(B));
        d_C = (float *)(mxGPUGetData(C));
        blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
        dc_sm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, W);
    }
    else if(ndims_A==4)
    {
        const mwSize *dims_A = mxGPUGetDimensions(A);
        const size_t M = dims_A[0];
        const size_t N = dims_A[1];
        const size_t L = dims_A[3];
        const mwSize dims_B[] = {M, N, 1, L};
        const mwSize dims_C[] = {M, N, 3, L};
        B = mxGPUCreateGPUArray(4, dims_B, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        C = mxGPUCreateGPUArray(4, dims_C, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_B = (float *)(mxGPUGetData(B));
        d_C = (float *)(mxGPUGetData(C));
        blocksPerGrid = (M*N*L + threadsPerBlock - 1) / threadsPerBlock;
        dc_mm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, L, W);
    }
    else
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);
    plhs[1] = mxGPUCreateMxArrayOnGPU(C);
    
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
}