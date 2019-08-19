# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cpu_nn.nn import Network


def read_data(data_path):
    # raw_data = np.genfromtxt(data_path, delimiter=',', dtype=None)
    raw_data = pd.read_csv(data_path, header=None)

    return raw_data


def convert_label_to_binary(data_df, primary_label, label_col_index=-1):
    data_df.iloc[:, label_col_index] = np.where(data_df.iloc[:, label_col_index] == primary_label, 1, 0)
    return data_df


def split_train_test(raw_data, train_fraction = 0.8, seed = None):
    # Shuffling the data
    raw_data = raw_data.sample(frac=1, random_state=seed)

    train_data = raw_data.iloc[:int(raw_data.shape[0]*train_fraction), :]
    test_data = raw_data.iloc[int(raw_data.shape[0]*train_fraction):, :]

    train_data_x = train_data.iloc[:, :-1].values.T
    train_data_y = train_data.iloc[:, -1].values.reshape(1, train_data.shape[0])
    test_data_x = test_data.iloc[:, :-1].values.T
    test_data_y = test_data.iloc[:, -1].values.reshape(1, test_data.shape[0])

    print("train_data_x:", train_data_x.shape)
    print("train_data_y", train_data_y.shape)
    print("test_data_x", test_data_x.shape)
    print("test_data_y", test_data_y.shape)

    return train_data_x, train_data_y, test_data_x, test_data_y


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, batch_size=None):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    FEATURE_SIZE = X.shape[0]

    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    network_obj = Network(layers_dims, seed=1, debug=True)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = network_obj.forward_propagation(X)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = network_obj.cost_cross_entropy(Y, AL)
        ### END CODE HERE ###

        # Backward propagation.
        grads = network_obj.backward_propagation(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        network_obj.update_parameters(gradients=grads, learning_rate=learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


if __name__ == "__main__":
    raw_data = read_data("../data/iris.data")
    raw_data = convert_label_to_binary(raw_data, "Iris-setosa", label_col_index=4)
    train_data_x, train_data_y, test_data_x, test_data_y = split_train_test(raw_data)

    L_layer_model(train_data_x[:, :2], train_data_y[:, :2], [train_data_x.shape[0], 3, 2, 1], num_iterations=3, print_cost=True)
