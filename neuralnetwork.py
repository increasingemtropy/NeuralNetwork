import numpy as np


class NeuralNetwork:
    
    """
    Neural Network class based on
    http://neuralnetworksanddeeplearning.com/
    """

    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

        self.num_layers = len(layer_sizes)
        self.activations = np.asarray([np.zeros(size) for size in layer_sizes])

    def feedforward(self, sample):
        """
        Feed forward through the network
        
        input vector of samples (which are themselves vectors)
        
        return vector of activations
        """

        # the input layer is the first activation layer
        self.activations[0] = sample

        # forward propagation
        for i in range(self.num_layers - 1):
            # z is 'weighted input'
            z = np.matmul(self.weights[i], self.activations[i]) + self.biases[i]
            self.activations[i + 1] = self.activation_func(z)

        # the output layer is the final activation layer
        return self.activations[-1]


    def print_accuracy(self, samples, labels):
        predictions = [self.feedforward(sample) for sample in samples]
        num_correct = sum([np.argmax(p) == np.argmax(l) for p, l in zip(predictions, labels)])
        print("{0}/{1} accuracy: {2}%".format(num_correct, len(samples), (num_correct / len(samples)) * 100))


    @staticmethod
    def activation_func(z):
        return sigmoid(z)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



