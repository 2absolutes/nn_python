__global__ void vector_matrix_addition(const float *matrix, const float *vector, float *out_matrix,
                                        int nrows, int ncols, int broadcast_dimension, int debug)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index;

    int vector_index;
    if (broadcast_dimension == 0){
        vector_index = col;
    }
    else if (broadcast_dimension == 1){
        vector_index = row;
    }

    if(row < nrows && col < ncols)
    {
        index = ncols * row + col;
        out_matrix[index] = matrix[index] + vector[vector_index];

        if(debug)
            printf("Row: %d, Column: %d, Cell_sum: %f, index: %d, vector_index: %d\n", row, col,
            out_matrix[index], index, vector_index);
    }
}