#include "cuda_runtime.h"
#include "cuda.h"
#include "cutil.h"
#include "cbcl_model.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <Magick++.h>
#include <exception>
#include <algorithm>
#include <assert.h>


using namespace std;
using namespace Magick;
#define PIXDEPTH MAGICKCORE_QUANTUM_DEPTH
typedef unsigned char uchar_t;

void write_image(const char* name,float* pimg,int wt,int ht);
void read_image(const char* name,float** ppimg,int * pwt,int* pht);
void gpu_filter(float* pimg,int wt,int ht,float* pout);
void cpu_to_gpu(band_info* pcin,int num_bands,band_info** ppcout,int copy=1);
void gpu_release_images(band_info** ppbands,int num_bands);
void gpu_to_cpu(band_info* pcin,int num_bands,band_info** ppcout,int copy=1);
void callback_c1_baseline(band_info*,int, band_info*,int,band_info** ,int* );
void callback_c2_baseline(band_info*,int,band_info* ,int,band_info*,int,band_info**,int*);
void callback_c2b_baseline(band_info*,int,band_info* ,int,band_info*,int,float**,int*);

void cpu_create_c0(float* pimg,int width,int height,band_info** ppc,int* pbands);
void cpu_load_filters(const char* filename,band_info** ppfilt,int* pnfilts);
void cpu_release_images(band_info** ppbands,int num_bands);

void write_image(const char* name,float* pimg,int wt,int ht)
{
	/*create blank image*/
	Image out_img(Geometry(wt,ht),ColorGray(0));
	/*scale the image*/
	uchar_t* pbuff		  = new uchar_t[ht*wt];
	float min_val = *((float*)min_element(&pimg[0],&pimg[ht*wt]));
	float max_val = *((float*)max_element(&pimg[0],&pimg[ht*wt]));
	int	  count	  = ht*wt;
	for(int i=0;i<count;i++)
		pbuff[i]=255*(pimg[i]-min_val)/(max_val-min_val+1e-5);
	/*write*/
	out_img.depth(8);
	out_img.type(MagickLib::GrayscaleType);
	out_img.getPixels(0,0,wt,ht);
	out_img.modifyImage();
	out_img.readPixels(MagickLib::GrayQuantum,pbuff);
	/*save changes*/
	out_img.syncPixels();
	out_img.write(name);
	delete [] pbuff;
}



void read_image(const char* name,float** ppimg,int * pwt,int* pht)
{
	Image in_img;
	in_img.read(name);
	uchar_t* pbuff = NULL;
	int height = in_img.rows();
	int width  = in_img.columns();

	*pht   = height;
	*pwt   = width;
	pbuff  = new uchar_t[height*width];
	*ppimg = new float[height*width];
	if(!*ppimg)
	{
		fprintf(stderr,"Unable to allocate image\n");
		throw int(-1);
	}
	in_img.read(name);
	/*change depth*/
	in_img.type(MagickLib::GrayscaleType);
	in_img.depth(8);
	/*read the pixels*/
	in_img.getPixels(0,0,width,height);
	in_img.writePixels(MagickLib::GrayQuantum,pbuff);
	/*transfer to double*/
	for(int i=0;i<height*width;i++)
		(*ppimg)[i]=(float)pbuff[i]/255.0;
	delete[] pbuff;
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
		read_image(argv[1],&pimg,&wt,&ht);
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
		//CUDA_SAFE_CALL(cudaMemcpyToArray(gpu_filt_array,0,0,phuge,28*7*4,cudaMemcpyHostToDevice));
		/*copy back*/
		float * pfilt = new float[49*4];
		cudaMemcpy2DFromArray(pfilt,28,gpu_filt_array,0,0,7*4,7*4,cudaMemcpyDeviceToHost);
		write_image("full-patch.jpeg",pfilt,7,28);
		delete [] pfilt;

		printf("computing s1\n");
	    unsigned int hTimer;
		CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
        CUT_SAFE_CALL( cutResetTimer(hTimer) );
        CUT_SAFE_CALL( cutStartTimer(hTimer) );
		callback_c1_baseline(c0,c0bands,patches_c0,num_patches,&s1,&num_s1);
		//callback_c2_baseline(c0,c0bands,patches_c0,num_patches,patches_c1,num_c1_patches,&s1,&num_s1);
		callback_c2b_baseline(c0,c0bands,patches_c0,num_patches,patches_c1,num_c1_patches,&pc2b,&nc2b);
        CUT_SAFE_CALL( cutStopTimer(hTimer) );
        double gpuTime = cutGetTimerValue(hTimer);
		printf("Time taken for s1: %lf\n",gpuTime);

#if 0
		for(int i=0;i<nc2b;i++)
		{
			cout<<pc2b[i]<<endl;
		}
#endif
#if 0
		printf("s1:%d\n",num_s1);
		printf("done with s1\n");
		for(int i=0;i<num_s1;i++)
		{
			char filename[256];
			sprintf(filename,"s1-%d.jpeg",i);
			write_image(filename,s1[i].ptr,s1[i].width,s1[i].height*s1[i].depth);
		}
		cpu_release_images(&s1,num_s1);
#endif
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