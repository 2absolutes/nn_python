"""
Gpu computation for common usages of functions used in a neural network. The input to the functions is expected to be in
 numpy which can be then copied to and from the device.
"""

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule


def matrix_multiplication(matrix1, matrix2):
    matrix1_rows = matrix1.shape[0]
    matrix1_cols = matrix1.shape[1]
    matrix2_rows = matrix2.shape[0]
    matrix2_cols = matrix2.shape[1]
    output_matrix = np.empty((matrix1_rows, matrix2_cols))

    ker_code = SourceModule(open("./kernels/matrix_multiplication.c", 'r').read())

    matmul = ker_code.get_function("matmul")


def scalar_scalar_addition():
    pass


def scalar_matrix_addition(scalar, matrix):
    return None


def vector_vector_addition(vector1, vector2):
    pass


def vector_matrix_addition():
    pass


def scalar_matrix_multiplication():
    pass


def element_wise_exponent(matrix, base="e"):
    pass


def element_wise_reciprocal(matrix, numerator=1.0):
    pass


def cross_entropy(vector1, vector2):
    """
    Place holder. Probably not much to gain by computing on gpu. Incomplete because low priority.
    :param vector1:
    :param vector2:
    :return:
    """
    pass
