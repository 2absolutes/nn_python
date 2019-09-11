__global__ void matrix_reciprocal(const float scalar, const float *matrix, float *out_matrix, int nrows, int ncols, int debug)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index;

    if (debug && row == 0 && col == 0){
        printf("Matrix: (%d, %d)\n", nrows, ncols);
    }

    if(row < nrows && col < ncols)
    {
        index = ncols * row + col;
        out_matrix[index] = scalar/matrix[index];

        if(debug)
            printf("Row: %d, Column: %d, Cell_sum: %f, index: %d\n", row, col, out_matrix[index], index);
    }
}
