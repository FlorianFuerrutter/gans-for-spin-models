#pragma once
//------------------

#define N_TERM   1e3	//updates until termalization
#define N_BINS   1e4	//actual data series size
#define N_SWEEPS 1e2	//sweeps between bins
		
#define LX 64
#define LY 64
#define N LX*LY

#define DATA_PATH "../data/train/"

//------------------

#if 0
	#define PRECISION float
#else
	#define PRECISION double
#endif 


