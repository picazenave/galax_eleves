#define GALAX_MODEL_GPU
#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	accelerationsGPU[i].x=0;
	accelerationsGPU[i].y=0;
	accelerationsGPU[i].z=0;

	for (int voisin = 0; voisin < n_particles; voisin++) //for each particule
        {
            
                float delta_pos_x = positionsGPU[voisin].x - positionsGPU[i].x;
                float delta_pos_y = positionsGPU[voisin].y - positionsGPU[i].y;
                float delta_pos_z = positionsGPU[voisin].z - positionsGPU[i].z;

                float delta_pos_x_sqr = delta_pos_x * delta_pos_x;
                float delta_pos_y_sqr = delta_pos_y * delta_pos_y;
                float delta_pos_z_sqr = delta_pos_z * delta_pos_z;

                float d_sqr = delta_pos_x_sqr + delta_pos_y_sqr + delta_pos_z_sqr;

                float d_cube = 1;
                if (d_sqr < 1.0f)
                {
                    d_cube=10;
                }
                else
                {
                    float d = sqrt(d_sqr);
                    d_cube = d * d * d;
                }

				accelerationsGPU[i].x+= delta_pos_x * DIFF_T * EPS * (1 / d_cube) * massesGPU[voisin];
				accelerationsGPU[i].y+= delta_pos_y * DIFF_T * EPS * (1 / d_cube) * massesGPU[voisin];
				accelerationsGPU[i].z+= delta_pos_z * DIFF_T * EPS * (1 / d_cube) * massesGPU[voisin];
            
        }
}


__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;

	positionsGPU[i].x+= velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y+= velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z+= velocitiesGPU[i].z * 0.1f;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}


#endif // GALAX_MODEL_GPU