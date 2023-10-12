#define GALAX_MODEL_GPU
#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, float *massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_particles)
		return;

	accelerationsGPU[i].x = 0;
	accelerationsGPU[i].y = 0;
	accelerationsGPU[i].z = 0;

	for (int voisin = 0; voisin < n_particles; voisin++) // for each particule
	{
		if (voisin != i)
		{
			float diffx = positionsGPU[voisin].x - positionsGPU[i].x;
			float diffy = positionsGPU[voisin].y - positionsGPU[i].y;
			float diffz = positionsGPU[voisin].z - positionsGPU[i].z;

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = sqrtf(dij);
				dij = 10.0 / (dij * dij * dij);
			}

			accelerationsGPU[i].x += diffx * (dij)*massesGPU[voisin];
			accelerationsGPU[i].y += diffy * (dij)*massesGPU[voisin];
			accelerationsGPU[i].z += diffz * (dij)*massesGPU[voisin];
		}
	}
	__syncthreads();
}

__global__ void maj_pos(float3 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;

	positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
	__syncthreads();
}

void update_position_cu(float3 *positionsGPU, float3 *velocitiesGPU, float3 *accelerationsGPU, float *massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks = (n_particles + (nthreads - 1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
	float3 r; // r_ij  [3 FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;										  // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2; // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube = 1.0f / sqrtf(distSixth); // s = m_j * invDistCube [1 FLOP]
	float s = bj.w * invDistCube;				 // a_i =  a_i + s * r_ij [6 FLOPS]
	ai.x += r.x * s;
	ai.y += r.y * s;
	ai.z += r.z * s;
	return ai;
}

#endif // GALAX_MODEL_GPU