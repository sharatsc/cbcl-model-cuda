#include "cuda.h"
#include "cutil.h"
#include <stdio.h>
#include <assert.h>
#include "cbcl_model.h"


#define BLOCK_SIZE 8 
using namespace std;

float* min_element(float* start,float* end)
{

	float* minptr=start;
	float  minval=*start;
	float  val   = minval;
	for(float* ptr=start;ptr!=end;ptr++)
	{
		val=*ptr;
		if(val<minval)
		{
			minval=val;
			minptr=ptr;
		}
	}
	return minptr;
}


float* max_element(float* start,float* end)
{
	float* maxptr=start;
	float  maxval=*start;
	float  val   = maxval;
	for(float* ptr=start;ptr!=end;ptr++)
	{
		val=*ptr;
		if(val>maxval)
		{
			maxval=val;
			maxptr=ptr;
		}
	}
	return maxptr;
}


/*
image texture
*/
texture<float,2,cudaReadModeElementType> teximg;
texture<float,2,cudaReadModeElementType> texfilt;

__host__ __device__ float* elptr(float* base,int depth,int row,int col,int height,int pitch)
{
	return (float*)((char*)base+depth*height*pitch+row*pitch)+col;
}

__global__ void kernel_c_local(float* dest,int depth,float wt,float ht,int poolxy,int stepxy,int srcwidth,int srcheight)
{
    int col       = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    int row       = blockIdx.y*BLOCK_SIZE+threadIdx.y; 
    int hpool     = poolxy/2;
    int x         = (col-1)*stepxy+hpool;
    int y         = (row-1)*stepxy+hpool;
    float xscale  = (float)srcwidth/(wt*stepxy+poolxy-1);
    float yscale  = (float)srcheight/(ht*stepxy+poolxy-1);
    if(row>=ht) return;
    if(col>=wt) return;
    //printf("Called with hpool:%d\n xscale:%f\n yscale:%f\n",hpool,xscale,yscale);
    //printf("(%d,%d),(%d,%d)->%d,%d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,row,col);
    for(int d=0;d<depth;d++)
    {
        float maxval  = 0;
        float pixval  = 0;
        for(int v=-hpool;v<hpool;v++)
            for(int u=-hpool;u<hpool;u++)
            {
                pixval=tex2D(teximg,(x+u)*xscale,d*srcheight+(y+v)*yscale);
                if(pixval>maxval)
                    maxval=pixval;
            }
        float* ptr = elptr(dest,d,row,col,ht,wt*sizeof(float));
        if(*ptr<maxval)
        *ptr = maxval;
    }
}

void gpu_c_local(
		IN  band_info* sin,     /*pointer to DEVICE storage*/
		IN  int in_bands,     /*number of input bands*/
		IN  int pool_xy,      /*spatial pooling: subsampling by pool_xy/2*/
		IN  int step_xy,      /*spatial subsampling factor*/
        IN  int pool_scale,   /*scale wise pooling: out_bands=in_bands/pool_scale*/
        IN  int step_scale,   /*scale incremenet step*/
		OUT band_info** c,   /*pointer to DEVICE storage*/
		OUT int* pout_bands,   /*number of output bands*/
        IN  bool copy
	)
{
   cudaArray*				imgarray;
   band_info*				h_outbands;
   float*                   d_ptr;
   cudaMemcpyKind           copydir=cudaMemcpyHostToDevice;
   int i,o,b;

   int out_bands = (in_bands-pool_scale)/step_scale+1;
   /*stage output*/
   h_outbands = new band_info[out_bands];
   assert(h_outbands!=NULL);
   for(i=0,o=0;i<=in_bands-pool_scale;i+=step_scale,o++)
   {
		h_outbands[o].height = (sin[i].height-pool_xy)/step_xy+1;
		h_outbands[o].width  = (sin[i].width-pool_xy)/step_xy+1;
		h_outbands[o].depth  = sin[i].depth;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_ptr,h_outbands[o].width*sizeof(float)*h_outbands[o].depth*h_outbands[o].height));
		CUDA_SAFE_CALL(cudaMemset(d_ptr,0,h_outbands[o].width*sizeof(float)*h_outbands[o].depth*h_outbands[o].height));
		h_outbands[o].pitch = h_outbands[o].width*sizeof(float);
		h_outbands[o].ptr   = d_ptr;
        h_outbands[o].where = ONDEVICE;
   }
   *pout_bands   = out_bands;
   *c            = h_outbands;

    cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
	CUDA_SAFE_CALL(cudaMallocArray(&imgarray,&imgdesc,sin[0].width,sin[0].height*sin[0].depth));
	/*bind the texture*/
	teximg.addressMode[0] = cudaAddressModeClamp;
	teximg.addressMode[1] = cudaAddressModeClamp;
	teximg.filterMode     = cudaFilterModeLinear; 
	teximg.normalized     = false;
		
   /*copy image*/ 
   for(i=0,o=0;i<=in_bands-pool_scale;i+=step_scale,o++)
   {
       for(b=0;b<pool_scale;b++)
       {
            copydir = cudaMemcpyHostToDevice;
            if(sin[i+b].where==ONDEVICE)
                copydir=cudaMemcpyDeviceToDevice;
	        CUDA_SAFE_CALL(cudaMemcpy2DToArray(imgarray,0,0,
										   sin[i+b].ptr,sin[i+b].width*sizeof(float),
										   sin[i+b].width*sizeof(float),sin[i+b].height*sin[i+b].depth,
									       copydir));
	        CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,imgarray));
	    	/*call the kernel*/
		    uint3 gridsz	 = make_uint3(ceilf((float)h_outbands[o].width/BLOCK_SIZE),ceilf((float)h_outbands[o].height/BLOCK_SIZE),1);
		    uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
		    kernel_c_local<<<gridsz,blocksz>>>(h_outbands[o].ptr,h_outbands[o].depth,h_outbands[o].width,h_outbands[o].height,pool_xy,step_xy,sin[i+b].width,sin[i+b].height);
		    CUDA_SAFE_CALL(cudaThreadSynchronize());
		    CUDA_SAFE_CALL(cudaUnbindTexture(teximg));
      }
   }
   /*copy image back to host*/
   for(b=0;copy && b<out_bands;b++)
   {
       int    sz  = h_outbands[b].height*h_outbands[b].width*h_outbands[b].depth;
       float* ptr = new float[sz];
       assert(ptr!=NULL);
       CUDA_SAFE_CALL(cudaMemcpy(ptr,h_outbands[b].ptr,sz*sizeof(float),cudaMemcpyDeviceToHost));
       CUDA_SAFE_CALL(cudaFree(h_outbands[b].ptr));
       h_outbands[b].ptr   =ptr;
       h_outbands[b].pitch =h_outbands[b].width*sizeof(float);
       h_outbands[b].where =ONHOST;
   }
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   CUDA_SAFE_CALL(cudaFreeArray(imgarray));
}


void cpu_release_images(band_info** ppbands,int num_bands)
{
	for(int i=0;i<num_bands;i++)
	{
        band_info* pb=*ppbands+i;
        if(pb->where==ONHOST)
		    delete[] pb->ptr;
        else
            CUDA_SAFE_CALL(cudaFree(pb->ptr));
	}
	delete [] *ppbands;
	*ppbands = NULL;
}

__global__ void kernel_resize(float *dest,int wt,int ht,float xscale,float yscale)
{
    int row=blockIdx.y*BLOCK_SIZE+threadIdx.y;
    int col=blockIdx.x*BLOCK_SIZE+threadIdx.x;
    if(row>=ht) return;
    if(col>=wt) return;
    dest[row*wt+col]=tex2D(teximg,(float)col*xscale,(float)row*yscale);
}

void gpu_create_c0(float* pimg,int width,int height,band_info** ppc,int* pbands,float scale,int num_scales,bool copy)
{
	*ppc				   = new band_info[num_scales];
	*pbands				   = num_scales;
    assert(*ppc!=NULL);
    assert(*pbands>=1);
    /*create first band*/
    (*ppc)->height           = height;
    (*ppc)->width            = width;
    (*ppc)->depth            = 1;
    (*ppc)->pitch            = width*sizeof(float);
    float* ptr               = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&ptr,height*width*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(ptr,pimg,height*width*sizeof(float),cudaMemcpyHostToDevice));
    (*ppc)->ptr              = ptr;
    assert((*ppc)->ptr);

    cudaArray*               pdimg;
    cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
    CUDA_SAFE_CALL(cudaMallocArray(&pdimg,&imgdesc,width,height));
	/*bind the texture*/
	teximg.addressMode[0] = cudaAddressModeClamp;
	teximg.addressMode[1] = cudaAddressModeClamp;
	teximg.filterMode     = cudaFilterModeLinear;
	teximg.normalized     = false;
	/*copy to array*/
	for(int b=1;b<num_scales;b++)
	{
        band_info* prev          = *ppc+b-1;
        band_info* pc            = *ppc+b;
		int bht			         = roundf(prev->height/scale);
		int bwt			         = roundf(prev->width/scale);
		pc->height		         = bht;
		pc->width		         = bwt;
		pc->pitch		         = bwt*sizeof(float);
		pc->depth		         = 1;
        /*map to texture*/
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(pdimg,0,0,
										   prev->ptr,prev->width*sizeof(float),
										   prev->width*sizeof(float),prev->height,
									       cudaMemcpyDeviceToDevice));
	    CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,pdimg));
        /*allocate the memory*/
        float* pdout             = NULL;
        CUDA_SAFE_CALL(cudaMalloc((void**)&pdout,bht*bwt*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemset(pdout,0,bht*bwt*sizeof(float)));
        pc->ptr                  = pdout;
        pc->where                = ONDEVICE;
	    /*call the kernel*/
		uint3 gridsz	 = make_uint3(ceilf((float)bwt/BLOCK_SIZE),ceilf((float)bht/BLOCK_SIZE),1);
		uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
		kernel_resize<<<gridsz,blocksz>>>(pdout,bwt,bht,(float)prev->width/bwt,(float)prev->height/bht);
        CUDA_SAFE_CALL(cudaThreadSynchronize());
        CUDA_SAFE_CALL(cudaUnbindTexture(teximg));						   
   }
   for(int b=0;copy && b<num_scales;b++)
   {
       band_info* pc =*ppc+b;
       float*     ptr=new float[pc->height*pc->width];
       CUDA_SAFE_CALL(cudaMemcpy(ptr,pc->ptr,pc->height*pc->width*sizeof(float),
                      cudaMemcpyDeviceToHost));
       CUDA_SAFE_CALL(cudaFree(pc->ptr));
       pc->ptr       = ptr;
       pc->where     = ONHOST;
   }
   CUDA_SAFE_CALL(cudaFreeArray(pdimg));
}

void cpu_create_c0(float* pimg,int width,int height,band_info** ppc,int* pbands,float scale,int num_scales)
{
	*ppc				   = new band_info[num_scales];
	*pbands				   = num_scales;
	assert(*ppc!=NULL);
    assert(*pbands>=1);
    /*create first band*/
    band_info*  prev         = *ppc;
    prev->height           = height;
    prev->width            = width;
    prev->depth            = 1;
    prev->pitch            = width*sizeof(float);
    prev->ptr              = new float[height*width];
    prev->where            = ONHOST;
    memcpy(prev->ptr,pimg,height*width*sizeof(float));
    assert(prev->ptr);
    /*determine dynamic range*/
    float minval           = *min_element(pimg,&pimg[height*width]);
    float maxval           = *max_element(pimg,&pimg[height*width]);
    float range            = maxval-minval+1e-6;

    /*create the other bands recursively*/
	for(int b=1;b<num_scales;b++,prev++)
	{
		pimg            = prev->ptr;
        width           = prev->width;
        height          = prev->height;
        
        band_info* pc	= *ppc+b;
		int bht			= roundf(height/scale);
		int bwt			= roundf(width/scale);
		pc->height		= bht;
		pc->width		= bwt;
		pc->pitch		= bwt*sizeof(float);
		pc->depth		= 1;
		pc->ptr			= new float[bht*bwt];
        pc->where       = ONHOST;
		assert(pc->ptr!=NULL);
        float cmin      = 1e6; /*current min*/
        float cmax      = 0;   /*current max*/
		for(int x=0;x<bwt;x++)
		{
			for(int y=0;y<bht;y++)
			{
				float sx = x*scale;
				float sy = y*scale;
				int   fx = floorf(sx); int  cx = ceilf(sx);cx=(cx>=width)?(width-1):cx;
				int   fy = floorf(sy); int  cy = ceilf(sy);cy=(cy>=height)?(height-1):cy;
				float xalpha=sx-fx;
				float yalpha=sy-fy;
				float val   =pimg[fx+fy*width]*(1-xalpha)*(1-yalpha)+
							 pimg[cx+fy*width]*(xalpha)*(1-yalpha)+
							 pimg[fx+cy*width]*(1-xalpha)*(yalpha)+
							 pimg[cx+cy*width]*(xalpha)*(yalpha);
				pc->ptr[y*bwt+x]=val;
                if(val<cmin) cmin=val;
                if(val>cmax) cmax=val;
			}
		}
        float crange = cmax-cmin+1e-6; 
        float factor = range/crange;
        for(int i=0;i<bht*bwt;i++)
            pc->ptr[i]=(pc->ptr[i]-cmin)*factor+minval;
	}
}


__global__  void kernel_s_norm_filter(float* dest,int pitch,int depth,int wt,int ht,int fwt,int fht)
{
    int         row             = blockIdx.y*BLOCK_SIZE+threadIdx.y;
    int         col             = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    int         xoff            = floorf(fwt/2);
    int         yoff            = floorf(fht/2);
	int u,v,d;
    float       den             = 0;
    float       num             = 0;
    float       pixval          = 0;
    float       filtval         = 0;
    if(row>ht-fht) return;
    if(col>wt-fwt) return;
    for(d=0;d<depth;d++)
    {
        int istride = d*ht;
        int fstride = d*fht;
        for(v=0;v<fht;v++)
            for(u=0;u<fwt;u++)
            {
                pixval =tex2D(teximg,col+u,istride+row+v);
                filtval=tex2D(texfilt,u,fstride+v);
                num    +=pixval*filtval;
                den    +=pixval*pixval;
            }
    }
    *elptr(dest,0,row+yoff,col+xoff,ht,pitch)=fabs(num)/sqrtf(den+1e-6);
}

/*
put the image into texture memory
put the filter into global memory
call the kernel for each band of the input (maybe change later)
*/
void gpu_s_norm_filter(band_info* cin,int in_bands,band_info* filt,int num_filt, band_info** pps, int *out_bands,bool copy)
{
   cudaArray*				imgarray;
   cudaArray*               filtarray;
   band_info*				h_outbands;
   float*					d_ptr;
   cudaMemcpyKind           copydir;
   /*channel description*/
   
   /*stage output*/
   h_outbands = new band_info[in_bands];
   for(int b=0;b<in_bands;b++)
   {
		h_outbands[b].height = cin[b].height;
		h_outbands[b].width  = cin[b].width;
		h_outbands[b].depth  = num_filt;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_ptr,cin[b].width*sizeof(float)*num_filt*cin[b].height));
		CUDA_SAFE_CALL(cudaMemset(d_ptr,0,cin[b].width*sizeof(float)*num_filt*cin[b].height));
		h_outbands[b].pitch = cin[b].width*sizeof(float);
        h_outbands[b].where = ONDEVICE;
		h_outbands[b].ptr   = d_ptr;
   }
   *pps      = h_outbands;
   *out_bands= in_bands;
	   
   /*copy image*/ 
   cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
   cudaChannelFormatDesc    filtdesc=cudaCreateChannelDesc<float>();
   CUDA_SAFE_CALL(cudaMallocArray(&filtarray,&filtdesc,filt[0].width,filt[0].height*filt[0].depth));
   CUDA_SAFE_CALL(cudaMallocArray(&imgarray,&imgdesc,cin[0].width,cin[0].height*cin[0].depth));
   /*fix address modes*/
    teximg.addressMode[0] = cudaAddressModeClamp;
    teximg.addressMode[1] = cudaAddressModeClamp;
    teximg.filterMode     = cudaFilterModePoint;
    teximg.normalized     = false;
    
    texfilt.addressMode[0] = cudaAddressModeClamp;
    texfilt.addressMode[1] = cudaAddressModeClamp;
    texfilt.filterMode     = cudaFilterModePoint;
    texfilt.normalized     = false;

    /*call the kernel*/
   for(int b=0;b<in_bands;b++)
   {
	    /*copy to array*/
        copydir     = cudaMemcpyHostToDevice;
        if(cin[b].where==ONDEVICE)
            copydir = cudaMemcpyDeviceToDevice;
		CUDA_SAFE_CALL(cudaMemcpy2DToArray(imgarray,0,0,
										   cin[b].ptr,cin[b].pitch,
										   cin[b].width*sizeof(float),cin[b].height*cin[b].depth,
									       copydir));
	    CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,imgarray));
        for(int f=0;f<num_filt;f++)
        {
            CUDA_SAFE_CALL(cudaMemcpy2DToArray(filtarray,0,0,
                                               filt[f].ptr,filt[f].width*sizeof(float),
                                               filt[f].width*sizeof(float),filt[f].height*filt[f].depth,
                                               cudaMemcpyHostToDevice));
	        CUDA_SAFE_CALL(cudaBindTextureToArray(texfilt,filtarray));
		    uint3 gridsz	 = make_uint3(ceilf((float)cin[b].width/BLOCK_SIZE),ceilf((float)cin[b].height/BLOCK_SIZE),1);
		    uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
            float* dest      = elptr(h_outbands[b].ptr,f,0,0,h_outbands[b].height,h_outbands[b].pitch);
		    kernel_s_norm_filter<<<gridsz,blocksz>>>(dest,h_outbands[b].pitch,cin[b].depth,cin[b].width,cin[b].height,filt[f].width,filt[f].height);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
	        CUDA_SAFE_CALL(cudaUnbindTexture(texfilt));
        }
        CUDA_SAFE_CALL(cudaUnbindTexture(teximg));
   }
   for(int b=0;copy && b<in_bands;b++)
   {
       int    sz  = h_outbands[b].height*h_outbands[b].width*num_filt;
       float* ptr = new float[sz];
       assert(ptr!=NULL);
       CUDA_SAFE_CALL(cudaMemcpy(ptr,h_outbands[b].ptr,sz*sizeof(float),cudaMemcpyDeviceToHost));
       CUDA_SAFE_CALL(cudaFree(h_outbands[b].ptr));
       h_outbands[b].ptr   =ptr;
       h_outbands[b].pitch =h_outbands[b].width*sizeof(float);
   }
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   /*copy image to output*/   
   CUDA_SAFE_CALL(cudaFreeArray(imgarray));
   CUDA_SAFE_CALL(cudaFreeArray(filtarray));
}


__global__  void kernel_s_rbf(float* dest,float sigma,int depth,int wt,int ht,int fwt,int fht)
{
    int         row             = blockIdx.y*BLOCK_SIZE+threadIdx.y;
    int         col             = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    int         xoff            = floorf(fwt/2);
    int         yoff            = floorf(fht/2);
    int         pitch           = wt*sizeof(float);
	int u,v,d;
    float       num             = 0;
    float       pixval          = 0;
    float       filtval         = 0;
    if(row>ht-fht) return;
    if(col>wt-fwt) return;
    for(d=0;d<depth;d++)
    {
        for(v=0;v<fht;v++)
            for(u=0;u<fwt;u++)
            {
                pixval =tex2D(teximg,col+u,d*ht+row+v);
                filtval=tex2D(texfilt,u,d*fht+v);
                num    += (pixval-filtval)*(pixval-filtval);
            }
    }
    /*
     *printf("%d,%d:%.6f,%0.6f\n",row,col,num,expf(-num/sigma));
     */
    *elptr(dest,0,row+yoff,col+xoff,ht,pitch)=exp(-num/sigma);
}

/*
put the image into texture memory
put the filter into global memory
call the kernel for each band of the input (maybe change later)
*/
void gpu_s_rbf(band_info* cin,int in_bands,band_info* filt,int num_filt, float sigma,band_info** pps, int *out_bands,bool copy)
{
   cudaArray*				imgarray;
   cudaArray*               filtarray;
   band_info*				h_outbands;
   float*					d_ptr;
   cudaMemcpyKind           copydir=cudaMemcpyHostToDevice;
   /*channel description*/
   
   /*stage output*/
   h_outbands = new band_info[in_bands];
   for(int b=0;b<in_bands;b++)
   {
		h_outbands[b].height = cin[b].height;
		h_outbands[b].width  = cin[b].width;
		h_outbands[b].depth  = num_filt;
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_ptr,cin[b].width*sizeof(float)*num_filt*cin[b].height));
		CUDA_SAFE_CALL(cudaMemset(d_ptr,0,cin[b].width*sizeof(float)*num_filt*cin[b].height));
		h_outbands[b].pitch = cin[b].width*sizeof(float);
		h_outbands[b].ptr   = d_ptr;
        h_outbands[b].where = ONDEVICE;
   }
   *pps      = h_outbands;
   *out_bands= in_bands;
	   
   /*copy image*/ 
   cudaChannelFormatDesc	imgdesc=cudaCreateChannelDesc<float>();
   cudaChannelFormatDesc    filtdesc=cudaCreateChannelDesc<float>();
   CUDA_SAFE_CALL(cudaMallocArray(&filtarray,&filtdesc,filt[0].width,filt[0].height*filt[0].depth));
   CUDA_SAFE_CALL(cudaMallocArray(&imgarray,&imgdesc,cin[0].width,cin[0].height*cin[0].depth));
   /*fix address modes*/
    teximg.addressMode[0] = cudaAddressModeClamp;
    teximg.addressMode[1] = cudaAddressModeClamp;
    teximg.filterMode     = cudaFilterModePoint;
    teximg.normalized     = false;
    
    texfilt.addressMode[0] = cudaAddressModeClamp;
    texfilt.addressMode[1] = cudaAddressModeClamp;
    texfilt.filterMode     = cudaFilterModePoint;
    texfilt.normalized     = false;

    sigma                  = 2*sigma*sigma;
    /*call the kernel*/
   for(int b=0;b<in_bands;b++)
   {
        copydir = cudaMemcpyHostToDevice;
        if(cin[b].where==ONDEVICE)
            copydir=cudaMemcpyDeviceToDevice;
	    /*copy to array*/
		CUDA_SAFE_CALL(cudaMemcpy2DToArray(imgarray,0,0,
										   cin[b].ptr,cin[b].pitch,
										   cin[b].width*sizeof(float),cin[b].height*cin[b].depth,
									       copydir));
	    CUDA_SAFE_CALL(cudaBindTextureToArray(teximg,imgarray));
        for(int f=0;f<num_filt;f++)
        {
            //printf("Processing S2: (%d,%d)\n",b,f);
            CUDA_SAFE_CALL(cudaMemcpy2DToArray(filtarray,0,0,
                                               filt[f].ptr,filt[f].width*sizeof(float),
                                               filt[f].width*sizeof(float),filt[f].height*filt[f].depth,
                                               cudaMemcpyHostToDevice));
	        CUDA_SAFE_CALL(cudaBindTextureToArray(texfilt,filtarray));
		    uint3 gridsz	 = make_uint3(ceilf((float)cin[b].width/BLOCK_SIZE),ceilf((float)cin[b].height/BLOCK_SIZE),1);
		    uint3 blocksz	 = make_uint3(BLOCK_SIZE,BLOCK_SIZE,1);
            float* dest      = elptr(h_outbands[b].ptr,f,0,0,h_outbands[b].height,h_outbands[b].pitch);
		    kernel_s_rbf<<<gridsz,blocksz>>>(dest,sigma,cin[b].depth,cin[b].width,cin[b].height,filt[f].width,filt[f].height);
            CUDA_SAFE_CALL(cudaThreadSynchronize());
	        CUDA_SAFE_CALL(cudaUnbindTexture(texfilt));
        }
        CUDA_SAFE_CALL(cudaUnbindTexture(teximg));
   }
   for(int b=0;copy && b<in_bands;b++)
   {
       int    sz  = h_outbands[b].height*h_outbands[b].width*num_filt;
       float* ptr = new float[sz];
       assert(ptr!=NULL);
       CUDA_SAFE_CALL(cudaMemcpy(ptr,h_outbands[b].ptr,sz*sizeof(float),cudaMemcpyDeviceToHost));
       CUDA_SAFE_CALL(cudaFree(h_outbands[b].ptr));
       h_outbands[b].ptr   =ptr;
       h_outbands[b].pitch =h_outbands[b].width*sizeof(float);
       h_outbands[b].where =ONHOST;
   }
   CUDA_SAFE_CALL(cudaThreadSynchronize());
   /*copy image to output*/   
   CUDA_SAFE_CALL(cudaFreeArray(imgarray));
   CUDA_SAFE_CALL(cudaFreeArray(filtarray));
}

void cpu_c_global(
	IN band_info* s,      /*pointer to device storage*/
	IN int in_bands,      /*number of input bands*/
	OUT float** ppc,          /*pointer to DEVICE storage*/
	OUT int* out_units   /*=input depth*/	
)
{
	*out_units = s[0].depth;
	*ppc       = new float[*out_units];
	assert(*ppc);
	
	float* pc  = *ppc;
	memset(pc,0,sizeof(float)*(*out_units));

	for(int d=0;d<s[0].depth;d++)
	{
		for(int b=0;b<in_bands;b++)
		{
			int    numel  = s[b].height*s[b].width;
			float* ptr    = s[b].ptr+d*numel;
			float* pmaxval= max_element(ptr,ptr+numel);
			pc[d]         = max(*pmaxval,pc[d]);
		}
	}
}


