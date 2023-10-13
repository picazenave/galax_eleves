#define GALAX_MODEL_GPU
#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include "helper_math.h"
#define DIFF_T (0.1f)
#define EPS (1.0f)

#define N_THREADS 32

__global__ void compute_acc(float4 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles)
		return;

	accelerationsGPU[i].x = 0;
	accelerationsGPU[i].y = 0;
	accelerationsGPU[i].z = 0;

	for (int voisin = 0; voisin < n_particles; voisin++) // for each particule
	{
		float4 diff = positionsGPU[voisin] - positionsGPU[i];

		float dij = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

		float temp = 10;
		float temp2 = rsqrtf(dij);
		temp2 = 10.0 * (temp2 * temp2 * temp2);
		dij = dij < 1.0 ? temp : temp2;

		// int dij_sup_1 = dij >= 1;
		// float dij_sqrt=sqrtf(dij);
		// dij=10/((dij_sup_1*dij_sqrt*dij_sqrt*dij_sqrt)+1-1*dij_sup_1);

		const float a = (dij)*positionsGPU[voisin].w;
		accelerationsGPU[i].x += diff.x * a;
		accelerationsGPU[i].y += diff.y * a;
		accelerationsGPU[i].z += diff.z * a;
	}
	/*


	float4 bi;
	bi.x = positionsGPU[i].x;
	bi.x = positionsGPU[i].y;
	bi.x = positionsGPU[i].z;
	bi.w = massesGPU[i];
	for (int voisin = 0; voisin < n_particles; voisin++) // for each particule
	{
		float4 bvoisin;
		bvoisin.x = positionsGPU[voisin].x;
		bvoisin.x = positionsGPU[voisin].y;
		bvoisin.x = positionsGPU[voisin].z;
		bi.w = massesGPU[voisin];
		accelerationsGPU[i] = bodyBodyInteraction(bi, bvoisin, accelerationsGPU[i]);

	}
	*/
}

__global__ void maj_pos(float4 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles)
		return;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;

	positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
}

void update_position_cu(float4 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, int n_particles)
{
	int nthreads = N_THREADS;
	int nblocks = (n_particles + (nthreads - 1)) / nthreads;

	calculate_forces<<<nblocks, nthreads>>>(positionsGPU, accelerationsGPU, n_particles);
	// compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU,accelerationsGPU, n_particles);
	maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

__device__ void bodyBodyInteraction(float4 bi, float4 bj, float3 &ai)
{
	float3 diff = {bj.x - bi.x, bj.y - bi.y, bj.z - bi.z};
	float diffIJ = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

	diffIJ = rsqrtf(diffIJ);
	diffIJ=diffIJ*diffIJ*diffIJ;
	diffIJ*=10.0;
	diffIJ=diffIJ<10.0?diffIJ:10.0;

	// int sup_a_un = truncf(diffIJ) ;
	// float dij_sqrt=rsqrtf(diffIJ);
	// diffIJ=(10*sup_a_un*dij_sqrt*dij_sqrt*dij_sqrt)+((1-sup_a_un)*10);


	const float a = diffIJ*bj.w;
	ai.x += diff.x * a;
	ai.y += diff.y * a;
	ai.z += diff.z * a;
}
__device__ void tile_calculation(float4 myPosition, float3 &accel)
{
	extern __shared__ float4 shPosition[N_THREADS];
	for (int i = 0; i < blockDim.x; i++)
	{
		bodyBodyInteraction(myPosition, shPosition[i], accel);
	}
}

__global__ void calculate_forces(float4 *globalX, float3 *globalA, int n_particles)
{
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid >= n_particles)
		return;

	extern __shared__ float4 shPosition[N_THREADS];
	float4 myPosition = globalX[gtid];
	int i, tile;
	float3 acc = {0.0f, 0.0f, 0.0f};

	for (i = 0, tile = 0; i < n_particles; i += blockDim.x, tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		shPosition[threadIdx.x] = globalX[idx];
		__syncthreads();
		tile_calculation(myPosition, acc);
		__syncthreads();
	} // Save the result in global memory for the integration step.
	globalA[gtid] = acc;
}

#endif // GALAX_MODEL_GPU