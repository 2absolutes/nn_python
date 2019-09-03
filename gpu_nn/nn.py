import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from gpu_nn import gpu_helper


# TODO: Current code architecture does not allow to set up each layers separately
# 1. Each layers can be initialized in a different way
# 2. Different activation functions can be used for different layers (will require work for both back and forward prop)
# 3. Support different solvers (eg: adam, sgd, batch gd, etc)

class Network(object):
    UPDATE_FLAG = False

    def __init__(self, sizes, seed=None, debug=False):
        """
        Setup the network by initializing the weights and biases. The input data is arranged column-wise,
        i.e each column is one input data.
        :param sizes: list of dimension size of each layer. Including the input and output layer.
        :param seed: can be set for reproducibility. Default in None, which will introduce randomness on each run.
        :param debug: can be set to True for debug level print statements. Default is False.
        """

        if seed is not None:
            np.random.seed(seed)

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []

        self._initialize_parameters()

        self._debug = debug

        if self._debug:
            print("\n\nInitialization Values")
            print("Layers: \n", sizes)
            print("\nWeights: \n\t length: {},\n\t values: \n\t {}".format(len(self.weights), self.weights))
            print("\nBiases: \n\t length: {},\n\t values: \n\t {}".format(len(self.biases), self.biases))

    def _initialize_parameters(self):
        """
        Initialize parameters for each layer.
        Initialize weights randomly from a standard gaussian distribution.
        The weights are scaled by 0.01.
        Biases are initialized to zeroes.
        """
        if self._debug:
            print("Initializing weights and biases ...")
        for layer in range(1, self.num_layers):
            self.biases.append(np.zeros((self.sizes[layer], 1)))
            self.weights.append(np.random.randn(self.sizes[layer], self.sizes[layer - 1]) * 0.01)

    @staticmethod
    def _one_layer_linear_forward(a_prev, l_weight, l_bias):
        """
        Compute the linear part of forward propagation (wx + b) for one layer
        :param a_prev: activations from previous layer. Will be input values for 1st hidden layer
        :param l_weight: weight params for current layer
        :param l_bias: biases for current layer
        :return: (l_weight * a) + l_bias, cache (components used to calculate z for this layer)
        """

        wx = gpu_helper.matrix_multiplication(l_weight, a_prev)
        z = gpu_helper.vector_vector_addition(wx, l_bias)
        cache = (a_prev, l_weight, l_bias)

        return z, cache

    @staticmethod
    def _activation_sigmoid(z):
        """
        sigmoid activation function
        :param z: linear part of the forward layer. Apply the sigmoid function on this.
        :return: The result of applying the sigmoid function on z
        """
        exp_z = gpu_helper.element_wise_exponent(z)
        a_denominator = gpu_helper.scalar_vector_addition(1.0, exp_z)
        a = gpu_helper.element_wise_reciprocal(a_denominator)

        return a
