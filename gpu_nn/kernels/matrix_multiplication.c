__global__ void matmul(const float *matrix1, const float *matrix2, float *out_matrix,
                        int matrix1_nrow, int matrix1_ncol, int matrix2_nrow, int matrix2_ncol)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < matrix1_nrow && col < matrix2_ncol)
    {
        float current_cell_sum = 0.0;
        for(int i = 0; i < matrix2_nrow; i++){
            current_cell_sum += matrix1[row * matrix1_nrow + i] * matrix2[matrix2_nrow * i + col];
        }
        out_matrix[row * matrix1_nrow + col] = current_cell_sum;
    }
}