# 学籍番号:21M30821
# 氏名:山上理久

## multi_gpu.cu(提出)
baseline.cuの4GPU版(f_nodeが4GPU構成のため)

###
```
N=2048: 0.014627 s (1174.534342 GFlops)

error: 4.240295
```

### 実行方法
```
module load cuda

nvcc -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp" -std=c++11 multi_gpu.cu 

./a.out
```


## baseline.cu
1GPU(shared memory)

### 性能
```
N=2048: 0.049202 s (349.172935 GFlops)

error: 0.000364
```

### 実行方法
```
module load cuda

nvcc -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp" -std=c++11 baseline.cu 

./a.out
```


## example.cpp
4プロセスMPI(5プロセスにするとerrorが大きくなる)

### 性能
```
N=256: 0.050008 s (0.670981 GFlops)

error: 0.000016
```

### 実行方法
```
module load intel-mpi

mpicxx example.cpp -std=c++11

mpirun -np 4 ./a.out
```