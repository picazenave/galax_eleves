#define GALAX_MODEL_GPU
#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

#define N_THREADS 128

__global__ void maj_pos(float4 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;

	positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
}

__device__ void bodyBodyInteraction(float4 bi, float4 bj, float3 &ai)
{
	float3 diff;
	diff.x = bj.x - bi.x;
	diff.y = bj.y - bi.y;
	diff.z = bj.z - bi.z;
	float diffIJ = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

	diffIJ = rsqrtf(diffIJ);
	diffIJ = diffIJ * diffIJ * diffIJ;
	diffIJ *= 10.0;

	diffIJ = diffIJ < 10.0 ? diffIJ : 10.0;

	float a = diffIJ * bj.w;
	ai.x += diff.x * a;
	ai.y += diff.y * a;
	ai.z += diff.z * a;
}

__device__ void tile_calculation(float4 myPosition, float3 &accel)
{
	int i;
	extern __shared__ float4 shPosition[N_THREADS];
	for (i = 0; i < blockDim.x; i++)
	{
		bodyBodyInteraction(myPosition, shPosition[i], accel);
	}
}

__global__ void calculate_forces(float4 *positionsGPU, float3 *accelerationsGPU, int n_particles)
{
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float4 shPosition[N_THREADS];
	float4 myPosition = positionsGPU[gtid];
	int i, tile;
	float3 acc = {0.0f, 0.0f, 0.0f};

	for (i = 0, tile = 0; i < n_particles; i += blockDim.x, tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		shPosition[threadIdx.x] = positionsGPU[idx];
		__syncthreads();
		tile_calculation(myPosition, acc);
		__syncthreads();
	} // Save the result in global memory for the integration step.
	accelerationsGPU[gtid] = acc;
}


void update_position_cu(float4 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, int n_particles)
{
	int nthreads = N_THREADS;
	int nblocks = (n_particles + (nthreads - 1)) / nthreads;

	calculate_forces<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	// compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU,accelerationsGPU, n_particles);
	maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU