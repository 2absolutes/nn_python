"""
Gpu computation for common usages of functions used in a neural network. The input to the functions is expected to be in
 numpy which can be then copied to and from the device.

TODO: Define block/grid size
TODO: Severe inefficiency suspected while copying to device and back for each step. CUDA streams can be used instead
"""

import os
import sys
import math
import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule

BLOCK_DIMS = (32, 32, 1)
GRID_DIMS = (1, 1, 1)

DIR_PATH = os.path.realpath("..")
sys.path.append(DIR_PATH)
print("dir vdswv", DIR_PATH)


def matrix_multiplication(matrix1, matrix2, debug=0):
    """

    :param matrix1:
    :param matrix2:
    :return:
    """
    matrix1_nrows = np.int32(matrix1.shape[0])
    matrix1_ncols = np.int32(matrix1.shape[1])
    matrix2_nrows = np.int32(matrix2.shape[0])
    matrix2_ncols = np.int32(matrix2.shape[1])

    if matrix1_ncols != matrix2_nrows:
        raise ValueError("Matrices cannot be multiplied due to dimension mismatch")

    output_matrix = np.empty((matrix1_nrows, matrix2_ncols)).astype(np.float32)

    print(f"Matrix1: {matrix1}, Matrix2: {matrix2}")

    matrix1_gpu = driver.mem_alloc(matrix1.nbytes)
    matrix2_gpu = driver.mem_alloc(matrix2.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix1_gpu, matrix1)
    driver.memcpy_htod(matrix2_gpu, matrix1)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_multiplication.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matmul = ker_code.get_function("matmul")

    matmul(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           matrix1_nrows, matrix1_ncols,
           matrix2_nrows, matrix2_ncols,
           np.int32(debug),
           block=BLOCK_DIMS,
           grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)
    return output_matrix


def scalar_matrix_addition(scalar, matrix, debug=0):
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/scalar_matrix_addition.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    scalar_matrix_add = ker_code.get_function("scalar_matrix_add")

    scalar_matrix_add(np.float32(scalar), matrix_gpu, output_matrix_gpu,
                      np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                      np.int32(debug),
                      block=BLOCK_DIMS,
                      grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)
    return output_matrix_cpu


def matrix_matrix_addition(matrix1, matrix2):
    """
    Addition of 2 2d matrices
    :param matrix1:
    :param matrix2:
    :return:
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Cannot perform addition. Shapes don't match. {} != {}".format(matrix1.shape, matrix2.shape))

    output_matrix = np.empty(matrix1.shape).astype(np.float32)

    matrix1_gpu = driver.mem_alloc(matrix1.nbytes)
    matrix2_gpu = driver.mem_alloc(matrix2.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix1_gpu, matrix1)
    driver.memcpy_htod(matrix2_gpu, matrix2)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_addition.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matadd = ker_code.get_function("matrix_addition")

    matadd(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           np.int32(matrix1.shape[0]), np.int32(matrix1.shape[1]),
           np.int32(0),
           block=BLOCK_DIMS,
           grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)

    return output_matrix


def vector_matrix_addition(vector, matrix):
    """
    Addition of a vector and a matrix. Vector is broadcasted.
    :param matrix:
    :param vector:
    :return:
    """

    if vector.shape[0] == 1:
        broadcast_dimension = 0
    elif vector.shape[1] == 1:
        broadcast_dimension = 1
    else:
        raise ValueError("Vector is not broadcastable. {} != {}".format(matrix.shape, vector.shape))

    output_matrix = np.empty(matrix.shape).astype(np.float32)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    vector_gpu = driver.mem_alloc(vector.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)
    driver.memcpy_htod(vector_gpu, vector)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/vector_matrix_addition.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    vector_matrix_add = ker_code.get_function("vector_matrix_addition")

    vector_matrix_add(matrix_gpu, vector_gpu, output_matrix_gpu,
                      np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                      np.int32(broadcast_dimension),
                      np.int32(0),
                      block=BLOCK_DIMS,
                      grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)

    return output_matrix


def scalar_matrix_multiplication(scalar, matrix):
    """

    :param scalar:
    :param matrix:
    :return:
    """
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_scalar_multiplication.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matrix_scalar_multiply = ker_code.get_function("matrix_scalar_multiplication")

    matrix_scalar_multiply(matrix_gpu, np.float32(scalar), output_matrix_gpu,
                           np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                           np.int32(0),
                           block=BLOCK_DIMS,
                           grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)
    return output_matrix_cpu


def element_wise_exponent(matrix, base=math.e):
    """
    Takes each element of the matrix and does base^element
    :param matrix:
    :param base:
    :return:
    """
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/exponent_matrix.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    exponent_matrix = ker_code.get_function("exponent_matrix")

    exponent_matrix(np.float32(base), matrix_gpu, output_matrix_gpu,
                    np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                    np.int32(0),
                    block=BLOCK_DIMS,
                    grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    return output_matrix_cpu


def element_wise_reciprocal(matrix, numerator=1.0):
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    numpy_answer_cpu = np.divide(numerator, matrix)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)
    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_reciprocal.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matrix_reciprocal = ker_code.get_function("matrix_reciprocal")

    matrix_reciprocal(np.float32(numerator), matrix_gpu, output_matrix_gpu,
                      np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                      np.int32(0),
                      block=BLOCK_DIMS,
                      grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)
    return output_matrix_cpu


def cross_entropy(vector1, vector2):
    """
    Place holder. Probably not much to gain by computing on gpu. Incomplete because low priority.
    :param vector1:
    :param vector2:
    :return:
    """
    pass


def matrix_element_multiplication(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Cannot perform addition. Shapes don't match. {} != {}".format(matrix1.shape, matrix2.shape))

    output_matrix = np.empty(matrix1.shape).astype(np.float32)

    matrix1_gpu = driver.mem_alloc(matrix1.nbytes)
    matrix2_gpu = driver.mem_alloc(matrix2.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix1_gpu, matrix1)
    driver.memcpy_htod(matrix2_gpu, matrix2)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_element_multiplication.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matrix_element_multiply = ker_code.get_function("matrix_element_multiplication")

    matrix_element_multiply(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           np.int32(matrix1.shape[0]), np.int32(matrix1.shape[1]),
           np.int32(0),
           block=BLOCK_DIMS,
           grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)

    return output_matrix


def matrix_subtraction(matrix1, matrix2):
    """
        Addition of 2 2d matrices
        :param matrix1:
        :param matrix2:
        :return:
        """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Cannot perform addition. Shapes don't match. {} != {}".format(matrix1.shape, matrix2.shape))

    output_matrix = np.empty(matrix1.shape).astype(np.float32)

    matrix1_gpu = driver.mem_alloc(matrix1.nbytes)
    matrix2_gpu = driver.mem_alloc(matrix2.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix1_gpu, matrix1)
    driver.memcpy_htod(matrix2_gpu, matrix2)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_subtraction.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matrix_subtract = ker_code.get_function("matrix_subtraction")

    matrix_subtract(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
                    np.int32(matrix1.shape[0]), np.int32(matrix1.shape[1]),
                    np.int32(0),
                    block=BLOCK_DIMS,
                    grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)

    return output_matrix


def matrix_element_division(matrix1, matrix2):
    """
        Division of 2 2d matrices
        :param matrix1:
        :param matrix2:
        :return:
        """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Cannot perform addition. Shapes don't match. {} != {}".format(matrix1.shape, matrix2.shape))

    output_matrix = np.empty(matrix1.shape).astype(np.float32)

    matrix1_gpu = driver.mem_alloc(matrix1.nbytes)
    matrix2_gpu = driver.mem_alloc(matrix2.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix1_gpu, matrix1)
    driver.memcpy_htod(matrix2_gpu, matrix2)

    kernel_file = os.path.join(DIR_PATH, "gpu_nn/kernels/matrix_element_division.c")
    ker_code = SourceModule(open(kernel_file, 'r').read())

    matrix_subtract = ker_code.get_function("matrix_element_division")

    matrix_subtract(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
                    np.int32(matrix1.shape[0]), np.int32(matrix1.shape[1]),
                    np.int32(0),
                    block=BLOCK_DIMS,
                    grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)

    return output_matrix
