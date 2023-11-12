//  ################ UNUSED BUT PRESERVED #####################

/* __kernel void initVectors(__global float *A) {	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int szx = get_global_size(0);
	int szy = get_global_size(1);
	int szz = get_global_size(2);
	float res = 0;
	// const int idx = x + y * szx + z * szx * szy;
	const int idx = x * szy * szz + y * szz + z;
	
	// printf("this: %d\n", y);
	if (z==0)
	{
		res = (float) x / ((float)y + 1.0);

	}
	else if (z==1)
	{
		res = 1.00;
		// printf("this: %d%d%d\n", x, y, z);

	}
	else
	{
		res = (float)y/((float)x + 1.00);
	}
	A[idx] = res;
}
 */
//  ################ UNUSED BUT PRESERVED #####################
 __kernel void initVectors(__global float *matr1) {	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int szx = get_global_size(0);
	int szy = get_global_size(1);
	int szz = get_global_size(2);
	float res = 0;
	const int idx0 = x + y * szx + 0 * szx * szy;
	const int idx1 = x + y * szx + 1 * szx * szy;
	const int idx2 = x + y * szx + 2 * szx * szy;
	matr1[idx0] = (float) x / ((float)y + 1.0);
	matr1[idx1] = 1.00;
	matr1[idx2] = (float)y/((float)x + 1.00);

}
__kernel void iterVectors(__global float *B) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int szx = get_global_size(0);
	int szy = get_global_size(1);
	int szz = get_global_size(2);
	// const int idx = x + y * szx + z * szx * szy;
	const int idx = x + y * szx + 1 * szx * szy;
	const int idx1 = x+1 + y * szx + 0 * szx * szy;
	const int idx2 = x-1 + y * szx + 2 * szx * szy;
	float res = B[idx];
	// printf("doing\n");
	// if (z == 1 )
	// printf("this: %d\n", szx);
	if( x > 0 && x < szx-1)
	{
		// printf("doing\n");
		if (z == 1)
		{
			// printf("doing\n");
			for (int t=0;t<24;t++)
			{
				// printf("doing\n");
				res += rsqrt(B[idx1]+B[idx2]);
				// res += 1 / sqrt(B[idx1]+B[idx2]);
			}
			// if (x==1) printf("at %d and %d this: %f\n",x,y, res);
			// printf("at %d and %d this: %f\n",x,y, A[idx2]);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		B[idx] = res;
	}
	
}