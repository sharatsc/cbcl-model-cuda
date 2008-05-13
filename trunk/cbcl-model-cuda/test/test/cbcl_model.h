#ifndef __cbcl_model_h__
#define __cbcl_model_h__
#define IN 
#define OUT

typedef struct{
  int    __align__(8) height;
  int    __align__(8) width;
  int    __align__(8) depth;  
  int    __align__(8) pitch;
  float  __align__(8) *ptr;
}band_info;

	 void gpu_s_norm_filter(
		IN  band_info	*pc,     /*pointer to the DEVICE storage*/
		IN  int			in_bands,  /*number of input bands [IN]*/
		IN  band_info	*pfilt,  /*pointer to DEVICE storage*/
		IN  int			num_filt,  /*number of filtes=output depth*/
		OUT band_info	**pps,      /*pointer ot DEVICE storage*/
		OUT int			*pout_bands/*number of output bands*/
	   );


	 void gpu_s_exp_tuning(
		IN  band_info	*pc,     /*pointer to the DEVICE storage*/
		IN  int			in_bands,  /*number of input bands [IN]*/
		IN  band_info	*pfilt,  /*pointer to DEVICE storage*/
		IN  int			num_filt,  /*number of filtes=output depth*/
		OUT band_info	**pps,      /*pointer ot DEVICE storage*/
		OUT int			*pout_bands/*number of output bands*/
	   );

void gpu_c_generic(
		IN  band_info* s,     /*pointer to DEVICE storage*/
		IN  int in_bands,     /*number of input bands*/
		IN  int pool_xy,      /*spatial pooling: subsampling by pool_xy/2*/
		IN  int pool_scale,   /*scale wise pooling: out_bands=in_bands/pool_scale*/
		OUT band_info** c,      /*pointer to DEVICE storage*/
		OUT int* out_bands   /*number of output bands*/
	);

	void gpu_c_terminal(
		IN band_info* s,      /*pointer to device storage*/
		IN int in_bands,      /*number of input bands*/
		OUT int* out_units,   /*=input depth*/
		OUT float* c          /*pointer to DEVICE storage*/
		);

	void cpu_c_terminal(
	IN band_info* s,      /*pointer to device storage*/
	IN int in_bands,      /*number of input bands*/
	OUT float** ppc,          /*pointer to DEVICE storage*/
	OUT int* out_units   /*=input depth*/	
	);
#endif
