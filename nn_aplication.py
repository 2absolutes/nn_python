# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

from cpu_nn.nn import Network as network_cpu
from gpu_nn.nn import Network as network_gpu
from sklearn.neural_network import MLPClassifier


def read_data(data_path, header=None):
    # raw_data = np.genfromtxt(data_path, delimiter=',', dtype=None)
    raw_data = pd.read_csv(data_path, header=header)

    return raw_data


def convert_label_to_binary(data_df, primary_label, label_col_index=-1):
    data_df.iloc[:, label_col_index] = np.where(data_df.iloc[:, label_col_index] == primary_label, 1, 0)
    return data_df


def split_train_test(raw_data, train_fraction=0.8, seed=None):
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


def L_layer_model(X, Y,
                  layers_dims, learning_rate=0.99,
                  num_iterations=3000, batch_size=None,
                  computation="cpu",
                  print_cost=False,
                  seed=None):
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

    np.random.seed(seed)
    costs = []  # keep track of cost

    Network = network_cpu
    if computation == "gpu":
        Network = network_gpu
    network_obj = Network(layers_dims, seed=seed, debug=False)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        if batch_size is None:
            batch_size = X.shape[1]
        for batch in range(math.ceil(X.shape[1]/batch_size)):
            end = (batch + 1) * batch_size
            if end > X.shape[1]:
                end = X.shape[1]
            X_batch = X[:, batch * batch_size: end]
            Y_batch = Y[:, batch * batch_size: end]

            AL, caches = network_obj.forward_propagation(X_batch)
            cost = network_obj.cost_cross_entropy(Y_batch, AL)

            grads = network_obj.backward_propagation(AL, Y_batch, caches)

            network_obj.update_parameters(gradients=grads, learning_rate=learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        # if print_cost and i % 100 == 0:
        costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return network_obj

def sklearn_MLP(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
    classifier = MLPClassifier(hidden_layer_sizes=[2], solver="sgd", activation="relu",
                               learning_rate_init=learning_rate, max_iter=num_iterations, random_state=1, verbose=False)
    X_T = X.T
    Y_T = Y.T
    print(X_T.shape, Y_T.shape)
    Y_T = np.ravel(Y_T)
    print(X_T.shape, Y_T.shape)

    classifier.fit(X_T, Y_T)
    print(classifier.loss_)
    print(classifier.out_activation_)
    print(classifier.coefs_)
    print(classifier.intercepts_)
    return classifier


if __name__ == "__main__":
    seed = 1
    raw_data = read_data("../data/iris.data")
    primary_label = "Iris-setosa"

    # raw_data = read_data("../data/poker-hand-training-true.data", header=None)
    # print(raw_data[10].value_counts())
    # primary_label = 0

    raw_data = convert_label_to_binary(raw_data, primary_label)

    # raw_data = read_data("../data/sinx.csv", header=0)
    # raw_data.drop(raw_data.columns[[1,3,4]], axis=1, inplace=True)

    train_data_x, train_data_y, test_data_x, test_data_y = split_train_test(raw_data, seed=seed)

    feature_size = train_data_x.shape[0]
    trained_model = L_layer_model(train_data_x, train_data_y, [feature_size, 2, 1], num_iterations=10000,
                                  batch_size=10,
                                  computation="gpu",
                                  print_cost=True, seed=1)
    #
    y_hat, _ = trained_model.forward_propagation(test_data_x)
    y_hat_binary = np.where(y_hat > 0.5, 1, 0)
    #
    # trained_model = sklearn_MLP(train_data_x, train_data_y, [4, 100, 1], num_iterations=2000)
    # y_hat = trained_model.predict(train_data_x.T)
    # print(trained_model.score(test_data_x.T, test_data_y.T))
    #
    print(y_hat_binary.shape, test_data_y.shape)
    # print(y_hat > 0.5)
    print(y_hat[:, :20])
    print(y_hat_binary[:, :20])
    print(train_data_y[:, :20])
