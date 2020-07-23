import numpy as np
import neuralnetwork as nn
import matplotlib.pyplot as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

plt.imshow(training_images[0].reshape(28,28),cmap = 'gray')
plt.show()

# specify the layers, first layer should be 784 and output layer should be 10 for this example
layer_sizes_fwd = (784, 16, 16, 10)
layer_sizes_rev = (10, 16, 16, 784)

# number of images used for training, the rest is used for testing the performance
training_set_size = 10000


#get training images
training_set_images = training_images[:training_set_size]
training_set_labels = training_labels[:training_set_size]

test_set_images = training_images[training_set_size:]
test_set_labels = training_labels[training_set_size:]

# initialize neural network
net = nn.NeuralNetwork(layer_sizes_fwd)

# first training session
net.train_network(training_set_images, training_set_labels, 4, 10, 4.0)

# evaluate performance after first training session
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

# second training session
net.train_network(training_set_images, training_set_labels, 8, 20, 2.0)

# evaluate performance after second training session
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

# initialise reverse network
gen_net = nn.NeuralNetwork(layer_sizes_rev)

# first network provides some categorisation data
prediction_data = net.predict_test(training_set_images)

# train the generative network using the categorisation data from the first network
gen_net.train_network(prediction_data,training_set_images,4, 10, 4.0)

# calculate cost function just for funsies :)
gen_net.calculate_average_cost(prediction_data,training_set_images)

# second training session with slower learning rate 
gen_net.train_network(prediction_data,training_set_images,8, 20, 2.0)

# cost function again :)
gen_net.calculate_average_cost(prediction_data,training_set_images)

# not get the generative network to imagine what various numbers look like based on pure inputs
imagine = gen_net.predict_test(training_set_labels)

# uncomment and use this to plot the imagined digits
for y in range(0,18):
    plt.imshow(imagine[y].reshape(28,28),cmap = 'gray')
    plt.show()
    plt.imshow(training_set_images[y].reshape(28,28),cmap = 'gray')
    plt.show()