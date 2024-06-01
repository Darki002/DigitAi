import numpy

from network import Network

if __name__ == '__main__':
    sizes = [2, 4, 3]
    net = Network(sizes)
    r = net.feedforward([1, 1])
