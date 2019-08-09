import numpy as np

# TODO: Current code architecture does not allow to set up each layers separately
# 1. Each layers can be initialized in a different way
# 2. Different activation functions can be used for different layers


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
        :return: (l_weight * a) + l_bias
        """

        z = np.dot(l_weight, a_prev) + l_bias

        return z

    @staticmethod
    def _activation_sigmoid(z):
        """
        sigmoid activation function
        :param z: linear part of the forward layer. Apply the sigmoid function on this.
        :return: The result of applying the sigmoid function on z
        """
        a = 1/(1.0 + np.exp(-z))

        return a

    def _one_layer_forward_propagation(self, a_prev, l_weight, l_bias):
        """
        One pass of forward propagation across the network.
        :param a_prev: activations from previous layer
        :param l_weight: weight parameters of this layer
        :param l_bias: bias parameters of this layer
        :return: the activation from this layer
        """
        z = self._one_layer_linear_forward(a_prev, l_weight, l_bias)

        # TODO: control activation function by an argument. Different layers can have different activation functions
        a = self._activation_sigmoid(z)

        if self._debug:
            print("weights: {}, biases: {}, z: {}, a: {}".format(l_weight, l_bias, z, a))

        return a

    def forward_propagation(self, x):
        """
        One pass of forward propagation across the network (all layers).
        :param x: one batch of input data, shape: (input_size, num_of_samples)
        :return: the final activation from the last layer
        """
        a = x  # Assigning the input to a for reusability in the for loop below below
        for l_weight, l_bias in zip(self.weights, self.biases):
            a = self._one_layer_forward_propagation(a, l_weight, l_bias)

        return a

    @staticmethod
    def _cost_cross_entropy(y, y_h):
        """
        Computes the cross entropy loss between the real and predicted values
        :param y: real ("ground truth") values, shape (1, batch_size)
        :param y_h: predicted values (as a result of the feedforward process), shape (1, batch_size)
        :return: cross entropy loss
        """

        batch_size = y.shape[1]

        cost = (-1 * 1/batch_size) * sum([yi * np.log(y_hi) + (1-yi)*np.log(1-y_hi) for yi, y_hi in zip(y, y_h)][0])
        return cost


if __name__ == "__main__":
    network_obj = Network(sizes=[4, 3, 2, 1], seed=0, debug=True)
