#include "cuda_runtime.h"
#include "cuda.h"
#include "cutil.h"
#include "cbcl_model.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <exception>
#include <algorithm>
#include <assert.h>


using namespace std;
typedef unsigned char uchar_t;

/*utility function declarations*/
void cpu_write_image(const char* name,float* pimg,int wt,int ht);
void cpu_read_image(const char* name,float** ppimg,int * pwt,int* pht);
void cpu_load_filters(const char* filename,band_info** ppfilt,int* pnfilts);

/*call different layers with default options*/
void callback_c1_baseline(band_info*,int, band_info*,int,band_info** ,int* );
void callback_c2_baseline(band_info*,int,band_info* ,int,band_info*,int,band_info**,int*);
void callback_c2b_baseline(band_info*,int,band_info* ,int,band_info*,int,float**,int*);


void cpu_write_image(const char* name,float* pimg,int wt,int ht)
{
	/*scale the image*/
	uchar_t* pbuff= new uchar_t[ht*wt];
	float min_val = *((float*)min_element(&pimg[0],&pimg[ht*wt]));
	float max_val = *((float*)max_element(&pimg[0],&pimg[ht*wt]));
	int	  count	  = ht*wt;
	for(int i=0;i<count;i++)
		pbuff[i]=255*(pimg[i]-min_val)/(max_val-min_val+1e-5);
	/*write*/
    cutSavePGMub(name,pbuff,wt,ht);
	delete [] pbuff;
}



void cpu_read_image(const char* name,float** ppimg,int * pwt,int* pht)
{
	uchar_t* pbuff = NULL;
	unsigned int height = 0;
	unsigned int width  = 0;
    /*call cutil function to load PGM*/
    cutLoadPGMub(name,&pbuff,&width,&height);
	*pht   = height;
	*pwt   = width;
	*ppimg = new float[height*width];
	if(!*ppimg)
	{
		fprintf(stderr,"Unable to allocate image\n");
		throw int(-1);
	}
	for(int i=0;i<height*width;i++)
		(*ppimg)[i]=(float)pbuff[i]/255.0;

    free(pbuff);/*mixing deallocation methods:(*/	
	return;
}


int main(int argc,char* argv[])
{
	band_info* c0;
	band_info* copyc0;
	band_info* gpu_c0;
	int		   c0bands;

	band_info* patches_c0;
	band_info* patches_c1;
	int		   num_patches;
	int		   num_c1_patches;

	band_info* copy_patches;
	band_info* gpu_patches;
	band_info* s1;
	cudaArray* gpu_filt_array;
	int		   num_s1;

	float * pc2b;
	int     nc2b;

	//CUT_DEVICE_INIT();
	if(argc<3)
	{
		fprintf(stderr,"usage is %s <in img> <out img>\n",argv[0]);
		return -1;
	}
	try
	{
		float*	pimg;
		float*  pout;
		float*  phuge;
		int		ht	=	0;
		int		wt	=	0;
		cpu_read_image(argv[1],&pimg,&wt,&ht);
		cpu_create_c0(pimg,wt,ht,&c0,&c0bands);
		cpu_load_filters("patches_gabor.txt",&patches_c0,&num_patches);
		cpu_load_filters("patches_c1.txt",&patches_c1,&num_c1_patches);

		cudaChannelFormatDesc	filtdesc=cudaCreateChannelDesc<float>();
		CUDA_SAFE_CALL(cudaMallocArray(&gpu_filt_array,&filtdesc,7,28));
		phuge = new float[49*4];
		for(int f=0;f<num_patches;f++)
		{	
			cudaMemcpy2DToArray(gpu_filt_array,0,f*7,patches_c0[f].ptr,28,7*4,7,cudaMemcpyHostToDevice);
			memcpy((void*)(phuge+f*49),patches_c0[f].ptr,49*sizeof(float));
		}
		/*test texture*/
		/*copy back*/
		float * pfilt = new float[49*4];
		cudaMemcpy2DFromArray(pfilt,28,gpu_filt_array,0,0,7*4,7*4,cudaMemcpyDeviceToHost);
		cpu_write_image("full-patch.jpeg",pfilt,7,28);
		delete [] pfilt;

		printf("computing s1\n");
	    unsigned int hTimer;
        CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
        CUT_SAFE_CALL( cutResetTimer(hTimer) );
        CUT_SAFE_CALL( cutStartTimer(hTimer) );
		callback_c1_baseline(c0,c0bands,patches_c0,num_patches,&s1,&num_s1);  
		callback_c2b_baseline(c0,c0bands,patches_c0,num_patches,patches_c1,num_c1_patches,&pc2b,&nc2b);
        CUT_SAFE_CALL( cutStopTimer(hTimer) );
        double gpuTime = cutGetTimerValue(hTimer);
        printf("Time taken for s1: %lf\n",gpuTime);

		cpu_release_images(&c0,c0bands);
		cpu_release_images(&patches_c0,num_patches);
		delete[] pimg;
	}
	catch(...)
	{
		cout<<"Exception"<<endl;
	}
	printf("done...");
	fflush(stdin);
	getchar();
	return 0;
	
}
