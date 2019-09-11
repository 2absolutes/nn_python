"""
Gpu computation for common usages of functions used in a neural network. The input to the functions is expected to be in
 numpy which can be then copied to and from the device.

TODO: Define block/grid size
TODO: Severe inefficiency suspected while copying to device and back for each step. CUDA streams can be used instead
"""

import os
import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule

BLOCK_DIMS = (32, 32, 1)
GRID_DIMS = (1, 1, 1)


def matrix_multiplication(matrix1, matrix2):
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

    matrix1_gpu = driver.mem_alloc(matrix1.nbytes)
    matrix2_gpu = driver.mem_alloc(matrix2.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix.nbytes)

    driver.memcpy_htod(matrix1_gpu, matrix1)
    driver.memcpy_htod(matrix2_gpu, matrix1)
    print(f"CWD: {os.getcwd()}")
    ker_code = SourceModule(open("../gpu_nn/kernels/matrix_multiplication.c", 'r').read())

    matmul = ker_code.get_function("matmul")

    matmul(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           matrix1_nrows, matrix1_ncols,
           matrix2_nrows, matrix2_ncols,
           np.int32(0),
           block=BLOCK_DIMS,
           grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)
    return output_matrix


def scalar_matrix_addition(scalar, matrix):
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)

    ker_code = SourceModule(open("./kernels/scalar_matrix_addition.c", 'r').read())

    scalar_matrix_add = ker_code.get_function("scalar_matrix_add")

    scalar_matrix_add(scalar, matrix_gpu, output_matrix_gpu,
                      np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                      np.int32(0),
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

    ker_code = SourceModule(open("./kernels/matrix_addition.c", 'r').read())

    matadd = ker_code.get_function("matadd")

    matadd(matrix1_gpu, matrix2_gpu, output_matrix_gpu,
           np.int32(matrix1.shape[0]), np.int32(matrix1.shape[1]),
           np.int32(0),
           block=BLOCK_DIMS,
           grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix, output_matrix_gpu)

    return output_matrix


def matrix_scalar_multiplication(scalar, matrix):
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    print(matrix, scalar)
    numpy_answer_cpu = np.multiply(scalar, matrix)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)

    print(matrix.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_scalar_multiplication.c", 'r').read())

    matrix_scalar_multiply = ker_code.get_function("matrix_scalar_multiplication")

    matrix_scalar_multiply(matrix_gpu, scalar, output_matrix_gpu,
                                 np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                                 np.int32(1),
                                 block=BLOCK_DIMS,
                                 grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)


def element_wise_exponent(matrix, base="e"):
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

    ker_code = SourceModule(open("./kernels/exponent_matrix.c", 'r').read())

    exponent_matrix = ker_code.get_function("exponent_matrix")

    exponent_matrix(base, matrix_gpu, output_matrix_gpu,
                    np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                    np.int32(1),
                    block=BLOCK_DIMS,
                    grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)

    return output_matrix_cpu


def element_wise_reciprocal(matrix, numerator=1.0):
    output_matrix_cpu = np.empty(matrix.shape).astype(np.float32)

    print(matrix, numerator)
    numpy_answer_cpu = np.divide(numerator, matrix)

    matrix_gpu = driver.mem_alloc(matrix.nbytes)
    output_matrix_gpu = driver.mem_alloc(output_matrix_cpu.nbytes)

    driver.memcpy_htod(matrix_gpu, matrix)

    print(matrix.shape)
    print(output_matrix_cpu.shape)

    ker_code = SourceModule(open("./kernels/matrix_reciprocal.c", 'r').read())

    matrix_reciprocal = ker_code.get_function("matrix_reciprocal")

    matrix_reciprocal(numerator, matrix_gpu, output_matrix_gpu,
                      np.int32(matrix.shape[0]), np.int32(matrix.shape[1]),
                      np.int32(1),
                      block=BLOCK_DIMS,
                      grid=GRID_DIMS)

    driver.memcpy_dtoh(output_matrix_cpu, output_matrix_gpu)


def cross_entropy(vector1, vector2):
    """
    Place holder. Probably not much to gain by computing on gpu. Incomplete because low priority.
    :param vector1:
    :param vector2:
    :return:
    """
    pass
