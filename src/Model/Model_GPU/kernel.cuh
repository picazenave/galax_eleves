#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>

__global__ void calculate_forces(float4 *globalX, float3 *globalA, int n_particles);
__device__ void tile_calculation(float4 myPosition, float3 &accel);
void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles);
__device__ void bodyBodyInteraction(float4 bi, float4 bj, float3 &ai);
#endif

#endif // GALAX_MODEL_GPU
