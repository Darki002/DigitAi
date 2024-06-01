import random

import numpy as np


class Network:
    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.add(np.dot(w, a), b))
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        :param training_data: the training data (x, y) where x is the data and y the desired outcome.
        :param epochs: how many epochs to train
        :param mini_batch_size: size of the mini bathces
        :param eta: learning rate
        :param test_data: the data to evaluate the learning
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batchs = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')

    def update_mini_batch(self, mini_batch, eta):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_new_b, delta_new_w = self.backprop(x, y)
            new_b = [np.add(nb, dnb) for nb, dnb in zip(new_b, delta_new_b)]
            new_w = [np.add(nw, dnw) for nw, dnw in zip(new_w, delta_new_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, new_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, new_w)]

    def backprop(self, x, y):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.add(np.dot(w, activation), b)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))
