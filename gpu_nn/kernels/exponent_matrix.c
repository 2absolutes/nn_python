__global__ void exponent_matrix(const float base, const float *matrix, float *out_matrix, int nrows, int ncols, int debug)
{
//    #include <math.h>
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index;

    if (debug && row == 0 && col == 0){
        printf("Matrix: (%d, %d)\n", nrows, ncols);
    }

    if(row < nrows && col < ncols)
    {
        index = ncols * row + col;
        out_matrix[index] = pow(base, matrix[index]);
//        out_matrix[index] = exp(matrix[index]);

        if(debug)
            printf("Row: %d, Column: %d, Cell_value: %f, index: %d\n", row, col, out_matrix[index], index);
    }
}
