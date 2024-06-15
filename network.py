import pickle
import random
import numpy as np


class Network:
    def __init__(self, sizes: list[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def load():
        with open('network.pkl', 'rb') as f:
            return pickle.load(f)

    def save(self):
        with open('network.pkl', 'wb') as f:
            pickle.dump(self, f)

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        :param training_data: the training data (x, y) where x is the data and y the desired outcome.
        :param epochs: how many epochs to train
        :param mini_batch_size: size of the mini batches
        :param eta: learning rate
        :param test_data: the data to evaluate the learning
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
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
            new_b = [nb + dnb for nb, dnb in zip(new_b, delta_new_b)]
            new_w = [nw + dnw for nw, dnw in zip(new_w, delta_new_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, new_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, new_w)]

    def backprop(self, x, y):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])

        new_b[-1] = delta
        new_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sd = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sd

            new_b[-l] = delta
            new_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return new_b, new_w

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_result)


def cost_derivative(output_activation, y):
    return output_activation - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
