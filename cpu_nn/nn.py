import numpy as np

# TODO: Current code architecture does not allow to set up each layers separately
# 1. Each layers can be initialized in a different way
# 2. Different activation functions can be used for different layers (will require work for both back and forward prop)


class Network(object):

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
            print("Layers: \n", sizes)
            print("\nWeights: \n", self.weights)
            print("\nBiases: \n", self.biases)

    def _initialize_parameters(self):
        """
        Initialize parameters for each layer.
        Initialize weights randomly from a standard gaussian distribution.
        The weights are scaled by 0.01.
        Biases are initialized to zeroes.
        """
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

        z = np.dot(l_weight, a_prev) + l_bias
        cache = (a_prev, l_weight, l_bias)

        return z, cache

    @staticmethod
    def _activation_sigmoid(z):
        """
        sigmoid activation function
        :param z: linear part of the forward layer. Apply the sigmoid function on this.
        :return: The result of applying the sigmoid function on z
        """
        a = 1/(1.0 + np.exp(-z))

        return a

    @staticmethod
    def _activation_relu(z):
        """
        sigmoid activation function
        :param z: linear part of the forward layer. Apply the sigmoid function on this.
        :return: The result of applying the relu function on z
        """
        a = np.max(0, z)

        return a

    def _one_layer_forward_propagation(self, a_prev, l_weight, l_bias, activation="sigmoid"):
        """
        One pass of forward propagation across one layer. i.e g(wx = b) where g is the activation function
        :param a_prev: activations from previous layer
        :param l_weight: weight parameters of this layer
        :param l_bias: bias parameters of this layer
        :param activation: the activation function to use for this layer
        :return: the activation from this layer
        """
        z, l_linear_cache = self._one_layer_linear_forward(a_prev, l_weight, l_bias)

        # TODO: Support more activation functions.
        if activation == "sigmoid":
            a = self._activation_sigmoid(z)
        elif activation == "relu":
            a = self._activation_relu(z)
        else:
            raise ValueError("{} activation function is not supported!".format(activation))

        if self._debug:
            print("weights: {}, biases: {}, z: {}, a: {}".format(l_weight, l_bias, z, a))

        l_cache = (l_linear_cache, z)

        return a, l_cache

    def forward_propagation(self, x):
        """
        One pass of forward propagation across the network (all layers).
        :param x: one batch of input data, shape: (input_size, num_of_samples)
        :return: the final activation from the last layer

        TODO: Support different activation functions for different layers
        """
        a = x  # Assigning the input to "a" for reusability in the for loop below below
        caches = []
        for l_weight, l_bias in zip(self.weights, self.biases):
            a, cache = self._one_layer_forward_propagation(a, l_weight, l_bias)
            caches.append(cache)

        return a, caches

    @staticmethod
    def cost_cross_entropy(y, y_h):
        """
        Computes the cross entropy loss between the real and predicted values
        :param y: real ("ground truth") values, shape (1, batch_size)
        :param y_h: predicted values (as a result of the feedforward process), shape (1, batch_size)
        :return: cross entropy loss
        """

        batch_size = y.shape[1]

        cost = (-1 * 1/batch_size) * sum([yi * np.log(y_hi) + (1-yi)*np.log(1-y_hi) for yi, y_hi in zip(y, y_h)][0])

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

        return cost

    @staticmethod
    def _one_layer_linear_backward(dz, cache):
        """
        Implements the linear part of the back propagation for one layer
        :param dz: derivative of the cost w.r.t the linear output (i.e Z) for this layer
        :param cache: tuple (a_prev, weight, bias) used to calculate z (during forward propagation) for this layer.
        :return:
        """
        a_prev, l_weight, l_bias = cache
        batch_size = a_prev.shpae[1]  # since the batch size is not explicitly store anywhere, we can get it like this

        dw = (1/batch_size) * np.matmul(dz, a_prev.T)
        db = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)
        da_prev = np.matmul(l_weight.T, dz)

        return da_prev, dw, db

    def _one_layer_backward_propagation(self, da, cache, activation="sigmoid"):
        """

        :param da:
        :param cache:
        :param activation:
        :return:
        """

        linear_cache, z = cache

        if activation == "sigmoid":
            dz = da * self._sigmoid_prime(z)
        else:
            raise ValueError("Activation function {} not supported".format(activation))

        da_prev, dw, db = self._one_layer_linear_backward(dz, linear_cache)

        return da_prev, dw, db

    def _sigmoid_prime(self, z):
        """
        Derivative of sigmoid function
        :param z: value to compute the sigmoid of and then take derivative w.r.t
        :return:
        """
        return self._activation_sigmoid(z) * (1 - self._activation_sigmoid(z))

    def backward_propagation(self, a_last, y, caches, cost_function="cross_entropy", da_last = None):
        """

        :param a_last:
        :param y:
        :param caches:
        :return:
        """
        gradients = [None] * self.num_layers
        batch_size = a_last.shape[1]
        y = y.reshape(a_last.shape)  # Just to make sure shapes of both a and y match

        if not da_last:
            if cost_function == "cross_entropy":
                da_last = -1 * (np.divide(y, a_last) + np.divide(1-y, 1-a_last))
            else:
                raise ValueError("{} cost function not supported".format(cost_function))

        l_cache = caches[self.num_layers - 1]
        l_da, l_dweight, l_dbias = self._one_layer_backward_propagation(da_last, l_cache)
        gradients[self.num_layers - 1] = (l_da, l_dweight, l_dbias)

        for layer in reversed((range(self.num_layers - 1))):
            l_cache = caches[layer]
            l_da, l_dweight, l_dbias = self._one_layer_backward_propagation(l_da, l_cache)
            gradients[layer] = (l_da, l_dweight, l_dbias)

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """

        :param gradients:
        :param learning_rate:
        :return:
        """
        for layer in range(self.num_layers):
            self.weights[layer] = self.weights[layer] - learning_rate*gradients[layer][1]
            self.biases[layer] = self.biases[layer] - learning_rate*gradients[layer][2]


if __name__ == "__main__":
    network_obj = Network(sizes=[4, 3, 2, 1], seed=0, debug=True)
