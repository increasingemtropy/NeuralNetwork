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
    
    # WORK IN PROGRESS - Run through network backwards to generate inputs
    # from outputs
    
    # def feedbackward(self, sample):
    #     """
    #     Feed backward through the network
        
    #     input vector of samples (which are themselves vectors)
        
    #     return vector of activations
    #     """

    #     # the input layer is the first activation layer
    #     self.activations[-1] = sample

    #     # forward propagation
    #     for i in range(self.num_layers - 1):
    #         # z is 'weighted input'
    #         z = np.matmul(self.weights[i], self.activations[i]) + self.biases[i]
    #         self.activations[i + 1] = self.activation_func(z)

    #     # the output layer is the final activation layer

    #     return self.activations[0]

    def print_accuracy_bool(self, samples, labels):
        # generate set of predictions for whether function is ON (>0.5) or OFF
        predictions = [self.feedforward(sample)>0.5 for sample in samples]
        num_correct = sum(p==l for p, l in zip(predictions, labels))
        print("{0}/{1} accuracy: {2}%".format(num_correct, len(samples), (num_correct / len(samples)) * 100))
    def print_accuracy(self, samples, labels):
        # generate set of predictions for whether function is ON (>0.5) or OFF
        predictions = [self.feedforward(sample) for sample in samples]
        num_correct = sum([np.argmax(p) == np.argmax(l) for p, l in zip(predictions, labels)])
        print("{0}/{1} accuracy: {2}%".format(num_correct, len(samples), (num_correct / len(samples)) * 100))
    
        
    def calculate_average_cost(self, samples, labels):
        predictions = self.feedforward(samples)
        average_cost = sum([self.cost_function(p, l) for p, l in zip(predictions, labels)]) / len(samples)
        print("average cost: {0}".format(average_cost))
    
    def predict_test(self, samples):
        predictions = [self.feedforward(sample) for sample in samples]
        return predictions
    
    def train_network(self, training_inputs, training_labels, generations, batch_size, learning_rate):
        """
        Training using stochastic gradient descent
        """
        # get training data into an array
        training_data = [(x, y) for x, y in zip(training_inputs, training_labels)]

        print("start training")
        for j in range(generations):
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]

            for batch in batches:
                self.update_batch(batch, learning_rate)

            print("generation {0} complete".format(j+1))

    def update_batch(self, batch, learning_rate):
        """
        Calculate gradient direction for a batch and then
        apply the change to the weights and biases
        """
        # these arrays will hold the change in direction for each weight and bias
        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]

        for sample, label in batch:
            # calculate the direction of the gradient for a single sample
            bias_deltas, weight_deltas = self.back_propagation(sample, label)

            # combine the result with results from previous samples
            bias_gradients = [b_gradient + b_delta for b_gradient, b_delta in zip(bias_gradients, bias_deltas)]
            weight_gradients = [w_gradient + w_delta for w_gradient, w_delta in zip(weight_gradients, weight_deltas)]

        # apply the found gradient to the weights and biases using learning_rate and the batch size
        self.biases = [b - (learning_rate / len(batch)) * b_gradient for b, b_gradient in zip(self.biases, bias_gradients)]
        self.weights = [w - (learning_rate / len(batch)) * w_gradient for w, w_gradient in zip(self.weights, weight_gradients)]

    def back_propagation(self, sample, label):
        """
        Feed a sample through the network and calculate the changes in weights and biases
        by propagating back through the network in such a way that the cost is minimized
        """
        bias_deltas = [np.zeros(b.shape) for b in self.biases]
        weight_deltas = [np.zeros(w.shape) for w in self.weights]

        # calculate the activations for all neurons by feeding a sample through the network
        self.feedforward(sample)

        # based on discussion by 3Blue1Brown: 
        # video: https://youtu.be/tIeHLnjs5U8
        L = -1

        # 'partial_deltas' is 2/3 rds of the chain rule
        # - the change in cost given a change in a(L) (= cost_function'(a(L), y))
        # - the change in a(L) given a change in z(L) (= sigmoid'(z(L)))
        partial_deltas = self.cost_function_derivative(self.activations[L], label) * \
                         self.activation_function_derivative(self.activations[L])

        # now multiply with the last 1/3rd of the chain rule, 
        # which is different for weights and biases
        # for biases: the change in z(L) given a change in b(L) = 1         
        # [5:46 in the video]
        # for weights: the change in z(L) given a change in w(L) = a(L-1)   
        # [5:11 in the video]
        bias_deltas[L] = partial_deltas  # * 1 (but this is redundant)
        weight_deltas[L] = np.dot(partial_deltas, self.activations[L - 1].T)

        # continue back propagation, stop at second to last layer, 
        # (we don't adjust input layer!)
        while L > -self.num_layers + 1:
            # to update the partial_deltas for a previous activation layer 
            # we need to multiply again with a third part
            # for the previous activation: the change in z(L) given a change in a(L-1) = w(L)   
            # [6:05 in the video]
            previous_layer_deltas = np.dot(self.weights[L].T, partial_deltas)

            # calculate 2/3rds chain rule for the biases and weights
            # and apply the third part separately
            partial_deltas = previous_layer_deltas * self.activation_function_derivative(self.activations[L - 1])
            bias_deltas[L - 1] = partial_deltas  # * 1 (but this is redundant)
            weight_deltas[L - 1] = np.dot(partial_deltas, self.activations[L - 2].T)

            L -= 1
            # can almost certainly be optimised
            # but it is implemented this way for clarity

        return bias_deltas, weight_deltas

    @staticmethod
    def activation_func(z):
        return sigmoid(z)
    
    @staticmethod
    def activation_function_derivative(a):
        """
        'a' is the activation, or in other words: a = sigmoid(z)
        (it's designed this way so we don't have to remember the 'z' when all we need is 'a' anyway)
        """
        return sigmoid_prime(a)

    """
    Cost function and its derivative
    These could be switched out for different functions?
    """

    @staticmethod
    def cost_function(output, y):
        return sum_of_squares(output, y)

    @staticmethod
    def cost_function_derivative(output, y):
        return sum_of_squares_prime(output, y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    derivative of sigmoid function is actually 'sigmoid(x) * (1 - sigmoid(x))'
    but if taking sigmoid(x) as input, we can use:
    """
    return x * (1 - x)


def sum_of_squares(output, y):
    return sum((a - b) ** 2 for a, b in zip(output, y))[0]


def sum_of_squares_prime(output, y):
    return 2 * (output - y)


