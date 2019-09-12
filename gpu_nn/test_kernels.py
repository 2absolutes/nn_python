import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

# -- initialize the device
import pycuda.autoinit

def test_matmul():

    # np.random.seed(123)
    max_dim = 10

    matrix1_nrows = np.int32(np.random.randint(max_dim))
    matrix1_ncols = np.int32(np.random.randint(max_dim))
    matrix2_nrows = matrix1_ncols
    matrix2_ncols = np.int32(np.random.randint(max_dim))

    matrix1_cpu = (np.random.randn(matrix1_nrows, matrix1_ncols)*1).astype(np.float32)
    matrix2_cpu = (np.random.randn(matrix2_nrows, matrix2_ncols)*1).astype(np.float32)
    output_matrix_cpu = np.empty((matrix1_nrows, matrix2_ncols)).astype(np.float32)

    numpy_answer_cpu = np.dot(matrix1_cpu, matrix2_cpu)

    # matrix1_gpu = gpuarray.to_gpu(matrix1_cpu)
    # matrix2_gpu = gpuarray.to_gpu(matrix2_cpu)
    # output_matrix_gpu = gpuarray.empty(output_matrix_cpu.shape, np.float32)

    matrix1_gpu = cuda.mem_alloc(matrix1_cpu.nbytes)
    matrix2_gpu = cuda.mem_alloc(matrix2_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix1_gpu, matrix1_cpu)
    cuda.memcpy_htod(matrix2_gpu, matrix2_cpu)

    print(matrix1_cpu.shape)
    print(matrix2_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_multiplication.c", 'r').read())

    matmul = ker_code.get_function("matmul")

    matmul(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           matrix1_nrows, matrix1_ncols,
           matrix2_nrows, matrix2_ncols,
           np.int32(0),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)



def test_matadd():

    np.random.seed(123)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))


    matrix1_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)
    matrix2_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)
    output_matrix_cpu = np.empty((nrows, ncols)).astype(np.float32)

    numpy_answer_cpu = matrix1_cpu + matrix2_cpu

    matrix1_gpu = cuda.mem_alloc(matrix1_cpu.nbytes)
    matrix2_gpu = cuda.mem_alloc(matrix2_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix1_gpu, matrix1_cpu)
    cuda.memcpy_htod(matrix2_gpu, matrix2_cpu)

    print(matrix1_cpu.shape)
    print(matrix2_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_addition.c", 'r').read())

    matmul = ker_code.get_function("matadd")

    matmul(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           nrows, ncols,
           np.int32(0),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


def test_scalar_matrix_add():

    np.random.seed(123)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))

    scalar = np.float32(np.random.random())
    matrix_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)
    output_matrix_cpu = np.empty(matrix_cpu.shape).astype(np.float32)

    numpy_answer_cpu = scalar + matrix_cpu

    matrix_gpu = cuda.mem_alloc(matrix_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    print(matrix_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/scalar_matrix_addition.c", 'r').read())

    scalar_matrix_add = ker_code.get_function("scalar_matrix_add")

    scalar_matrix_add(scalar, matrix_gpu, output_matrix_gpu,
           np.int32(matrix_cpu.shape[0]), np.int32(matrix_cpu.shape[1]),
           np.int32(0),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


def test_exponent_matrix():
    np.random.seed(123)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))

    scalar = np.abs(np.float32(np.random.random()))
    matrix_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)

    output_matrix_cpu = np.empty(matrix_cpu.shape).astype(np.float32)

    print(matrix_cpu, scalar)
    numpy_answer_cpu = np.power(scalar, matrix_cpu)

    matrix_gpu = cuda.mem_alloc(matrix_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    print(matrix_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/exponent_matrix.c", 'r').read())

    exponent_matrix = ker_code.get_function("exponent_matrix")

    exponent_matrix(scalar, matrix_gpu, output_matrix_gpu,
           np.int32(matrix_cpu.shape[0]), np.int32(matrix_cpu.shape[1]),
           np.int32(1),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


def test_matrix_reciprocal():
    np.random.seed(123)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))

    scalar = np.abs(np.float32(np.random.random()))
    matrix_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)

    output_matrix_cpu = np.empty(matrix_cpu.shape).astype(np.float32)

    print(matrix_cpu, scalar)
    numpy_answer_cpu = np.divide(scalar, matrix_cpu)

    matrix_gpu = cuda.mem_alloc(matrix_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    print(matrix_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_reciprocal.c", 'r').read())

    matrix_reciprocal = ker_code.get_function("matrix_reciprocal")

    matrix_reciprocal(scalar, matrix_gpu, output_matrix_gpu,
           np.int32(matrix_cpu.shape[0]), np.int32(matrix_cpu.shape[1]),
           np.int32(1),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


def test_matrix_scalar_multiplication():
    np.random.seed(123)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))

    scalar = np.abs(np.float32(np.random.random()))
    matrix_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)

    output_matrix_cpu = np.empty(matrix_cpu.shape).astype(np.float32)

    print(matrix_cpu, scalar)
    numpy_answer_cpu = np.multiply(scalar, matrix_cpu)

    matrix_gpu = cuda.mem_alloc(matrix_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    print(matrix_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_scalar_multiplication.c", 'r').read())

    matrix_scalar_multiplication = ker_code.get_function("matrix_scalar_multiplication")

    matrix_scalar_multiplication(matrix_gpu, scalar, output_matrix_gpu,
           np.int32(matrix_cpu.shape[0]), np.int32(matrix_cpu.shape[1]),
           np.int32(1),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


def test_vector_matrix_addition():

    np.random.seed(124)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))

    matrix_cpu = (np.random.randn(nrows, ncols) * 1).astype(np.float32)
    vector_cpu = (np.random.randn(nrows, 1) * 1).astype(np.float32)
    vector_row_cpu = (np.random.randn(nrows, 1) * 1).astype(np.float32)

    output_matrix_cpu = np.empty(matrix_cpu.shape).astype(np.float32)

    print(matrix_cpu, vector_cpu, vector_row_cpu)
    numpy_answer_cpu = np.add(vector_cpu, matrix_cpu)

    if vector_cpu.shape[0] == 1:
        broadcast_dimension = 0
    elif vector_cpu.shape[1] == 1:
        broadcast_dimension = 1
    else:
        raise ValueError("Vector is not broadcastable. {} != {}".format(matrix_cpu.shape, vector_cpu.shape))

    matrix_gpu = cuda.mem_alloc(matrix_cpu.nbytes)
    vector_gpu = cuda.mem_alloc(vector_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix_cpu)
    cuda.memcpy_htod(vector_gpu, vector_cpu)

    ker_code = SourceModule(open("./kernels/vector_matrix_addition.c", 'r').read())

    vector_matrix_add = ker_code.get_function("vector_matrix_addition")

    vector_matrix_add(matrix_gpu, vector_gpu, output_matrix_gpu,
                      np.int32(matrix_cpu.shape[0]), np.int32(matrix_cpu.shape[1]),
                      np.int32(broadcast_dimension),
                      np.int32(0),
                      block=(16, 16, 1),
                      grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


def test_matrix_element_multiplication():

    np.random.seed(123)
    max_dim = 10

    nrows = np.int32(np.random.randint(max_dim))
    ncols = np.int32(np.random.randint(max_dim))


    matrix1_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)
    matrix2_cpu = (np.random.randn(nrows, ncols)*1).astype(np.float32)
    output_matrix_cpu = np.empty((nrows, ncols)).astype(np.float32)

    numpy_answer_cpu = matrix1_cpu * matrix2_cpu

    matrix1_gpu = cuda.mem_alloc(matrix1_cpu.nbytes)
    matrix2_gpu = cuda.mem_alloc(matrix2_cpu.nbytes)
    output_matrix_gpu = cuda.mem_alloc(output_matrix_cpu.nbytes)

    cuda.memcpy_htod(matrix1_gpu, matrix1_cpu)
    cuda.memcpy_htod(matrix2_gpu, matrix2_cpu)

    print(matrix1_cpu.shape)
    print(matrix2_cpu.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_element_multiplication.c", 'r').read())

    matrix_element_multiplication = ker_code.get_function("matrix_element_multiplication")

    matrix_element_multiplication(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           nrows, ncols,
           np.int32(0),
           block=(16, 16, 1),
           grid=(1, 1, 1))

    cuda.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    print("\n\nGPU: \n", output_matrix_cpu, "\n\n")
    print("CPU: \n", numpy_answer_cpu)


if __name__ == "__main__":
    test_matmul()
    # test_matadd()
    # test_scalar_matrix_add()
    # test_exponent_matrix()
    # test_matrix_reciprocal()
    # test_matrix_scalar_multiplication()
    # test_vector_matrix_addition()
    # test_matrix_element_multiplication()