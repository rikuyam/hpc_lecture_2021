#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
using namespace std;

#define num_devs 4

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
      sum += A_s[k-ks] * B[N/2*k+j];
    }
  }
  C[N/2*i+j] = sum;
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

void errorcalc2(float *A, float *B, float *C, int N){
  #pragma omp parallel for
  for (int i=0; i<1024; i++)
    for (int k=0; k<2048; k++)
      for (int j=0; j<1024; j++)
        C[1024*i+j] -= A[2048*i+k] * B[1024*k+j];
  double err = 0;
  for (int i=0; i<1024; i++)
    for (int j=0; j<1024; j++)
      err += fabs(C[1024*i+j]);
  printf("error: %lf\n",err/1024/1024);
}

int main(int argc, char **argv) {
  int N = 2048;
  int M = 128;
  int size = N*N*sizeof(float);

  float A[2048*2048], B[2048*2048], C[2048*2048];
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }

  float subA[4][N/2*N], subB[4][N*N/2], subC[4][N/2*N/2];
  float *subA_d[4], *subB_d[4], *subC_d[4];
  for (int i=0; i<N/2; i++) {
    for (int j=0; j<N; j++) {
      subA[0][N*i+j] = A[N*i+j];
      subA[1][N*i+j] = A[N*i+j];
      subA[2][N*i+j] = A[N*(i+N/2)+j];
      subA[3][N*i+j] = A[N*(i+N/2)+j];
      subB[0][N/2*j+i] = B[N*j+i];
      subB[1][N/2*j+i] = B[N*j+i];
      subB[2][N/2*j+i] = B[N*j+(i+N/2)];
      subB[3][N/2*j+i] = B[N*j+(i+N/2)];
    }
  }
  for (int i=0; i<N/2; i++) {
    for (int j=0; j<N/2; j++) {
      subC[0][N/2*i+j] = 0;
      subC[1][N/2*i+j] = 0;
      subC[2][N/2*i+j] = 0;
      subC[3][N/2*i+j] = 0;
    }
  }

  cudaStream_t stream[4];
  for (int dev_id=0; dev_id<num_devs; dev_id++) {
    cudaSetDevice(dev_id);
    cudaStreamCreate(&stream[dev_id]);
    cudaMalloc(&(subA_d[dev_id]),size/2);
    cudaMalloc(&(subB_d[dev_id]),size/2);
    cudaMalloc(&(subC_d[dev_id]),size/4);
		cudaMemcpy(subA_d[dev_id],subA[dev_id],size/2,cudaMemcpyHostToDevice);
		cudaMemcpy(subB_d[dev_id],subB[dev_id],size/2,cudaMemcpyHostToDevice);
		cudaMemcpy(subC_d[dev_id],subC[dev_id],size/4,cudaMemcpyHostToDevice);
  }

  auto tic = chrono::steady_clock::now();
  for (int dev_id = 0; dev_id < num_devs; dev_id++) {
    cudaSetDevice(dev_id);
    dim3 grid(N/2/M, N/2);
    cudamatmul<<<grid,M,M*sizeof(float)>>>(subA_d[dev_id], subB_d[dev_id], subC_d[dev_id], N);
  }
  for (int dev_id = 0; dev_id < num_devs; dev_id++) {
    cudaStreamSynchronize(stream[dev_id]);
  }
  
  for (int dev_id = 0; dev_id < num_devs; dev_id++) {
    cudaSetDevice(dev_id);
    cudaMemcpy(subC[dev_id],subC_d[dev_id],size/4,cudaMemcpyDeviceToHost);
  }
  auto toc = chrono::steady_clock::now();

  for (int i=0; i<N/2; i++) {
    for (int j=0; j<N/2; j++) {
      C[N*i+j] = subC[0][N/2*i+j];
      C[N*i+(j+N/2)] = subC[1][N/2*i+j];
      C[N*(i+N/2)+j] = subC[2][N/2*i+j];
      C[N*(i+N/2)+(j+N/2)] = subC[3][N/2*i+j];
    }
  }
  double time = chrono::duration<double>(toc - tic).count();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);

  /*
  for (int dev_id=0; dev_id<num_devs; dev_id++) {
    errorcalc2(subA[dev_id],subB[dev_id],subC[dev_id],N);
  }
  */

  errorcalc(A,B,C,N);
  
  for (int dev_id=0; dev_id<num_devs; dev_id++) {
    cudaFree(subA_d[dev_id]);
    cudaFree(subB_d[dev_id]);
    cudaFree(subC_d[dev_id]);
  }
}
