__global__ void matmul(const float *matrix1, const float *matrix2, float *out_matrix,
                        int matrix1_nrow, int matrix1_ncol, int matrix2_nrow, int matrix2_ncol, int debug)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (debug && row == 0 && col == 0){
        printf("Matrix1: (%d, %d), Matrix2: (%d, %d)\n", matrix1_nrow, matrix1_ncol, matrix2_nrow, matrix2_ncol);
    }
    if(row < matrix1_nrow && col < matrix2_ncol)
    {

        float current_cell_sum = 0.0;
        for(int i = 0; i < matrix1_ncol; i++){
            if (debug)
            {
                printf("matrix1[%d] :: %f * matrix2[%d] :: %f = output[%d]%f\n",
                row * matrix1_ncol + i,
                matrix1[row * matrix1_ncol + i],
                i * matrix2_ncol + col,
                matrix2[i * matrix2_ncol + col],
                row * matrix2_ncol + col,
                matrix1[row * matrix1_ncol + i] * matrix2[i * matrix2_ncol + col]);
            }
            current_cell_sum += matrix1[row * matrix1_ncol + i] * matrix2[i * matrix2_ncol + col];
        }

        out_matrix[row * matrix2_ncol + col] = current_cell_sum;

        if(debug)
            printf("Row: %d, Column: %d, Cell_sum: %f, matrix2_ncol: %d, output_cell_index: %d\n", row, col, current_cell_sum, matrix2_ncol, row * matrix2_ncol + col);
    }
}