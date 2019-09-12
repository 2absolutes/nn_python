__global__ void matrix_subtraction(const float *matrix1, const float *matrix2, float *out_matrix, int nrows, int ncols, int debug)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index;

    if (debug && row == 0 && col == 0){
        printf("Matrix1: (%d, %d), Matrix2: (%d, %d)\n", nrows, ncols, nrows, ncols);
    }

    if(row < nrows && col < ncols)
    {
        index = ncols * row + col;
        out_matrix[index] = matrix1[index] - matrix2[index];

        if(debug)
            printf("Row: %d, Column: %d, Cell_sum: %f, index: %d\n", row, col, out_matrix[index], index);
    }
}