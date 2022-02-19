#pragma once
//------------------

#define N_TERM   1e4	//updates until termalization
#define N_BINS   1e2	//actual data series size
#define N_SWEEPS 1e3	//sweeps between bins
		
#define LX 16
#define LY 16
#define N LX*LY

#define DATA_PATH "../data/train/"

//------------------

#if 0
	#define PRECISION float
#else
	#define PRECISION double
#endif 


