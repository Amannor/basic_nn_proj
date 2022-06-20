import random
import typing
import math
import json

import numpy as np
from keras.datasets import mnist
import warnings
warnings.filterwarnings("error")


class Network(object):
#http://neuralnetworksanddeeplearning.com/chap1.html

    def __init__(self, sizes: typing.List[int]) -> None:
        weights_denominator_by_layer = {0: 1000, 1: 100}
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1)/100000 for y in sizes[1:]]


        self.weights = [np.random.randn(y, x)/weights_denominator_by_layer[i]
                        for i, (y, x) in enumerate(zip(sizes[1:], sizes[:-1]))]
        '''
        Note: to calculate layer i: a_i = activation_func(weights[i-1].dot(a_(i-1)+biases[i-1]))
        or simply put  func(wa+b)
        '''

    def feed_forward(self, a: np.ndarray) -> np.ndarray:
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)

        return a

    def SGD(self, training_data: typing.List[typing.Tuple[np.ndarray, int]], epochs: int, mini_batch_size: int, eta: float, test_data: typing.List[typing.Tuple[np.ndarray, int]] = None) -> None:
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch: typing.List[typing.Tuple[np.ndarray, int]], eta: float) -> None:
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x: np.ndarray, y) -> typing.Tuple[typing.List[np.ndarray]]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        # y_vec = np.zeros(len(self.biases[-1]))
        # y_vec[y] = 1
        cost_derivative_res = self.cost_derivative(activations[-1], get_one_hot(y)) #Instead of y should y_vec #TODO
        sigmoid_prime_res = sigmoid_prime(zs[-1])
        delta = cost_derivative_res * sigmoid_prime_res
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: typing.List[typing.Tuple[np.ndarray, int]]) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        final_layer = [self.feed_forward(x) for (x,y) in test_data]
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations: typing.List[np.ndarray], y) -> np.ndarray:
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def get_one_hot(index: int, size:int = 10) -> np.ndarray:
    res = np.zeros(size)
    res = res.reshape(10,1)
    res[index] = 1
    return res


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    Source: https://stackoverflow.com/a/57178527
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

# def sigmoid(x: np.ndarray) -> np.ndarray:
#     res = x
#     # See https://stackoverflow.com/a/30368735
#     try:
#         res = 1.0/(1.0+np.exp(-x))
#     except RuntimeWarning:
#         print("RuntimeWarning")
#         # import ipdb;
#         # ipdb.set_trace()
#     return res

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1.0-sigmoid(x))

def main() -> None:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X.shape = (train_X.shape[0], train_X.shape[1]*train_X.shape[2], 1)
    training_data = [(x, y) for x, y in zip(train_X, train_y)]
    # For debug only
    # training_data = [(x, get_one_hot(y)) for x, y in zip(train_X, train_y) if y == 2]
    # training_data = [(x, y) for x, y in zip(train_X, train_y) if y == 2]
    # training_data = training_data[:100] #For debug only


    test_X.shape = (test_X.shape[0], test_X.shape[1]*test_X.shape[2], 1)
    test_data = [(x, get_one_hot(y)) for x, y in zip(test_X, test_y)]

    net = Network([784, 30, 10])
    # net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
    for learning_rate in np.arange(0.5, 3.0, 0.5):
        print(f"*************Learning rate {learning_rate}*************")
        net.SGD(training_data, 30, 10, learning_rate, test_data=training_data)

if __name__ == '__main__':
    main()
