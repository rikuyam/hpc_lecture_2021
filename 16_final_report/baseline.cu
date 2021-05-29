#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
using namespace std;

__global__ void cudamatmul(float *A, float *B, float *C, int N) {
  int i = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
  for (int ks=0; ks<N; ks+=blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[N*i+ks+threadIdx.x];
    __syncthreads();
    for (int k=ks; k<ks+blockDim.x; k++) {
      sum += A_s[k-ks] * B[N*k+j];
    }
  }
  C[N*i+j] = sum;
}

void matmul(float *A, float *B, float *C, int N, int M){
  dim3 grid(N/M, N);
  auto tic = chrono::steady_clock::now();
  cudamatmul<<<grid,M,M*sizeof(float)>>>(A, B, C, N);
  cudaDeviceSynchronize();
  auto toc = chrono::steady_clock::now();
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
}

void errorcalc(float *A, float *B, float *C, int N){
  #pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  printf("error: %lf\n",err/N/N);
}

int main(int argc, char **argv) {
  int N = 2048;
  int M = 1024;
  if(argc==3){
    N = atoi(argv[1]);
    M = atoi(argv[2]);
  }
  int size = N * N * sizeof(float);
  float *A, *B, *C;
  cudaMallocManaged(&A, size);
  cudaMallocManaged(&B, size);
  cudaMallocManaged(&C, size);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }

  matmul(A,B,C,N,M);

  errorcalc(A,B,C,N);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
