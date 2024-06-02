import mnist_loader
from network import Network

if __name__ == '__main__':
    net = Network([784, 30, 10])
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.SGD(list(training_data), 30, 10, 3, list(test_data))
