#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define MPICHECK(cmd) do {                          \
  int res = cmd;                                    \
  if (res != MPI_SUCCESS) {                         \
    printf("MPI Failure at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t res = cmd;                            \
  if (res != cudaSuccess) {                         \
    printf("CUDA Failure at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("NCCL Failure at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {
  int size, rank, local_rank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  // Get local rank from Slurm to pick the correct GPU
  char* local_rank_str = getenv("SLURM_LOCALID");
  local_rank = local_rank_str ? atoi(local_rank_str) : 0;

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  size_t nElems = 1048576;

  if (rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaMalloc(&sendbuff, nElems * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, nElems * sizeof(float)));

  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, nElems, ncclFloat, ncclSum, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  std::cout << "Rank " << rank << " OK" << std::endl;

  // Cleanup
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  MPICHECK(MPI_Finalize());

  return 0;
}
