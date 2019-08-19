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
            print("\nWeights: \n\t length: {},\n\t values: \n\t {}".format(len(self.weights), self.weights))
            print("\nBiases: \n\t length: {},\n\t values: \n\t {}".format(len(self.biases), self.biases))

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
        a = 1.0/(1.0 + np.exp(-z))

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
            print("\nShapes:\nweights: {}, a_prev:{}, biases: {}, a:{}".format(l_weight.shape, a_prev.shape, l_bias.shape, a.shape))
            print("\nValues:\nweights: \n{}, \nbiases: \n{}, \nz: \n{}, \na: \n{}".format(l_weight, l_bias, z, a))

        l_cache = (l_linear_cache, z)

        return a, l_cache

    def forward_propagation(self, x):
        """
        One pass of forward propagation across the network (all layers).
        :param x: one batch of input data, shape: (input_size, num_of_samples)
        :return: the final activation from the last layer

        TODO: Support different activation functions for different layers
        """
        print("Check: W:\n{}, \nb:\n{}".format(self.weights, self.biases))
        a = x  # Assigning the input to "a" for reusability in the for loop below below
        caches = []
        for l_weight, l_bias in zip(self.weights, self.biases):
            a, cache = self._one_layer_forward_propagation(a, l_weight, l_bias)
            # if self._debug:
            print("A for this layer", a)
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
        batch_size = a_prev.shape[1]  # since the batch size is not explicitly store anywhere, we can get it like this

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

        :param cost_function:
        :param da_last:
        :param a_last:
        :param y:
        :param caches:
        :return:
        """
        hidden_layers = self.num_layers - 1
        gradients = [None] * hidden_layers
        batch_size = a_last.shape[1]
        y = y.reshape(a_last.shape)  # Just to make sure shapes of both a and y match

        if not da_last:
            if cost_function == "cross_entropy":
                da_last = -1 * (np.divide(y, a_last) + np.divide(1-y, 1-a_last))
            else:
                raise ValueError("{} cost function not supported".format(cost_function))

        l_cache = caches[hidden_layers - 1]
        l_da, l_dweight, l_dbias = self._one_layer_backward_propagation(da_last, l_cache)
        gradients[hidden_layers - 1] = (l_da, l_dweight, l_dbias)

        for layer in reversed((range(hidden_layers - 1))):
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
        if self._debug:
            print("Old Values: \nweights:{}, \nbiases:{}".format(self.weights, self.biases))

        for layer in range(self.num_layers - 1):
            self.weights[layer] = self.weights[layer] - learning_rate*gradients[layer][1]
            self.biases[layer] = self.biases[layer] - learning_rate*gradients[layer][2]

        if self._debug:
            print("New Values: \nweights:{}, \nbiases:{}".format(self.weights, self.biases))


if __name__ == "__main__":
    network_obj = Network(sizes=[4, 3, 2, 1], seed=0, debug=False)

    # ----- Check _one_layer_linear_forward()

    W = np.array([[ 1.74481176, -0.7612069,   0.3190391 ]])
    A = np.array([[ 1.62434536, -0.61175641],
                       [-0.52817175, -1.07296862],
                       [ 0.86540763, -2.3015387 ]])
    b = np.array([[-0.24937038]])
    print(network_obj._one_layer_linear_forward(A, W, b)[0])

    # ----- Check _one_layer_forward_propagation()

    A_prev = np.array([[-0.41675785, -0.05626683],
                       [-2.1361961,   1.64027081],
                      [-1.79343559, -0.84174737]])
    W = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
    b = np.array([[-0.90900761]])

    print(network_obj._one_layer_forward_propagation(A_prev, W, b, activation="sigmoid")[0])

    # ----- Check _one_layer_forward_propagation()
    X = np.array([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
         [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
         [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
         [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
         [ 0.07612761, -0.15512816,  0.63422534 , 0.810655  ]])

    W = [np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
                   [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
                   [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
                   [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]),
         np.array([[-0.12673638, -1.36861282, 1.21848065, -0.85750144],
                   [-0.56147088, -1.0335199, 0.35877096, 1.07368134],
                   [-0.37550472, 0.39636757, -0.47144628, 2.33660781]]),
         np.array([[0.9398248, 0.42628539, -0.75815703]])
         ]
    b = [np.array([[ 1.38503523],
                   [-0.51962709],
                   [-0.78015214],
                   [ 0.95560959]]),
         np.array([[ 1.50278553],
                   [-0.59545972],
                   [ 0.52834106]]), np.array([[-0.16236698]])]

    # print(W.shape, b.shape)

    network_obj.weights = W
    network_obj.biases = b
    print(network_obj.forward_propagation(X)[0])
