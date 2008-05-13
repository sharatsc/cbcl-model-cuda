#include "cuda.h"
#include "cutil.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <assert.h>
#include "cbcl_model.h"

void cpu_to_gpu(band_info* pcin,int num_bands,band_info** ppcout,int copy=1);
void gpu_to_cpu(band_info* pcin,int num_bands,band_info** ppcout,int copy=1);
void gpu_release_images(band_info** ppbands,int num_bands);
void callback_c1_baseline(band_info*,int, band_info*,int,band_info** ,int* );
void cpu_create_c0(float* pimg,int width,int height,band_info** ppc,int* pbands);
void cpu_load_filters(const char* filename,band_info** ppfilt,int* pnfilts);
__device__ float* elptr(float* base,int depth,int row,int col,int height,int pitch);


_