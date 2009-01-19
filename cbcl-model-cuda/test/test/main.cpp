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

#define TEST_CREATE_C0 0 
#define TEST_IO_FILTER 0
#define TEST_S1        1


using namespace std;
typedef unsigned char uchar_t;

/*utility function declarations*/
void cpu_write_image(const char* name,float* pimg,int wt,int ht);
void cpu_read_image(const char* name,float** ppimg,int * pwt,int* pht);
void cpu_read_filters(const char* filename,band_info** ppfilt,int* pnfilts);
void cpu_write_filters(band_info* pfilt,int nfilt,const char* filename);

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
    cout<<cutLoadPGMub(name,&pbuff,&width,&height);
    cout<<"loaded image:"<<width<<"x"<<height<<endl;
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


void cpu_read_filters(const char* filename,band_info** ppfilt,int* pnfilts)
{
	ifstream fin(filename);
	/*read number of filters*/
	int num_filters;

	fin>>num_filters;
	cout<<"Number of filters"<<num_filters<<endl;
	assert(num_filters >= 1);
	*pnfilts= num_filters;
	*ppfilt = new band_info[num_filters];
	assert(*ppfilt !=NULL);

	for(int i=0;i<num_filters;i++)
	{
		band_info* pfilt = *ppfilt+i;
		fin >> pfilt->depth;
		fin >> pfilt->height;
		fin >> pfilt->width;
		/*allocate memory for the image*/
		pfilt->pitch=pfilt->width*sizeof(float);
		pfilt->ptr  =new float[pfilt->depth*pfilt->height*pfilt->width];
		assert(pfilt->ptr);
		for(int d=0;d<pfilt->depth;d++)
		{
			float* ptr=pfilt->ptr+d*pfilt->height*pfilt->width;
			for(int y=0;y<pfilt->height;y++)
			{
				for(int x=0;x<pfilt->width;x++)
					fin>>ptr[y*pfilt->width+x];
			}
		}
	}
	fin.close();
}

void cpu_write_filters(band_info* pfilt,int nfilts,const char* filename)
{
    ofstream fout(filename);
    assert(nfilts>=1);
    assert(pfilt!=NULL);

    fout<<nfilts<<endl;

    for(int i=0;i<nfilts;i++,pfilt++)
    {
        fout<<pfilt->depth<<endl;
        fout<<pfilt->height<<endl;
        fout<<pfilt->width<<endl;

        for(int d=0;d<pfilt->depth;d++)
        {
            float* ptr=pfilt->ptr+d*pfilt->height*pfilt->width;
            for(int y=0;y<pfilt->height;y++)
            {
                for(int x=0;x<pfilt->width;x++)
                    fout<< *elptr(pfilt->ptr,d,y,x,pfilt->height,pfilt->pitch)<<" ";
                fout<<endl;
            }
        }
    }
    fout.close();
}

#if TEST_IO_FILTERS //test read_filters
int main(int argc,char* argv[])
{
    band_info* c0;
    int        nc0bands;
    cpu_read_filters("c0.txt",&c0,&nc0bands);
    cpu_write_filters(c0,nc0bands,"copyc0.txt");
    cpu_release_images(&c0,nc0bands);
    delete[] c0;
}
#elif TEST_CREATE_C0
int main(int argc,char* argv[])
{
    band_info *c0;
    int     nc0bands;
    float*  pimg;
    int     height;
    int     width;
	unsigned int hTimer;
    cpu_read_image("cameraman.pgm",&pimg,&width,&height);
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	cpu_create_c0(pimg,width,height,&c0,&nc0bands);
    double gpuTime = cutGetTimerValue(hTimer);
    printf("Time taken for C2: %lf\n",gpuTime);
    cpu_write_filters(c0,nc0bands,"camerman.txt");
    cpu_release_images(&c0,nc0bands);
    delete[] c0;
}
#elif TEST_S1
int main(int argc,char* argv[])
{
    band_info *c0;
    band_info *s1;
    band_info *c0patches;
    int     nc0patches;
    int     nc0bands;
    int     ns1bands;

    float*  pimg;
    int     height;
    int     width;
	unsigned int hTimer;
    cpu_read_image("cameraman.pgm",&pimg,&width,&height);
    cpu_read_filters("c0Patches.txt",&c0patches,&nc0patches);
	cpu_create_c0(pimg,width,height,&c0,&nc0bands,1.113,8);
    printf("Patches:%d\n",nc0patches);
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
    gpu_s_norm_filter(c0,nc0bands,c0patches,nc0patches,&s1,&ns1bands);
    double gpuTime = cutGetTimerValue(hTimer);
    printf("Time taken for S1: %lf\n",gpuTime);
    cpu_write_filters(c0,nc0bands,"c0.txt");
    cpu_write_filters(s1,ns1bands,"s1.txt");
    cpu_release_images(&c0,nc0bands);
    cpu_release_images(&s1,ns1bands);
    delete[] c0;
}
#else
int main(int argc,char* argv[])
{
	/*define storage for the layers*/
    band_info* c0;
	int		   c0bands;

    /*define storage for the prototypes/patches*/
	band_info* patches_c0;
	band_info* patches_c1;
	int		   num_c0;
	int		   num_c1;

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
		cpu_read_filters("patches_gabor.txt",&patches_c0,&num_c0);
		cpu_read_filters("patches_c1.txt",&patches_c1,&num_c1);

		printf("computing C2\n");
	    unsigned int hTimer;
        CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
        CUT_SAFE_CALL( cutResetTimer(hTimer) );
        CUT_SAFE_CALL( cutStartTimer(hTimer) );
		callback_c2b_baseline(c0,c0bands,patches_c0,num_c0,patches_c1,num_c1,&pc2b,&nc2b);
        CUT_SAFE_CALL( cutStopTimer(hTimer) );
        double gpuTime = cutGetTimerValue(hTimer);
        printf("Time taken for C2: %lf\n",gpuTime);

		cpu_release_images(&c0,c0bands);
		cpu_release_images(&patches_c0,num_c0);
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
#endif
