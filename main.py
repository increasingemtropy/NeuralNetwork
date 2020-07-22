import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt

# specify the layers, first layer should be 2 for inputs to boolean function, output is only a single layer
layer_sizes = (2,16,1)

# generate initial test data, 1000 pairs of values in [0,1]
test_data = np.random.random((1000,2,1))
# generate labels for the data, here we do XOR
# XOR   0  1
#     0 F  T
#     1 T  F
test_labels = np.random.random((1000,1,1))
for i in range(0,1000):
    test_labels[i,0]=(test_data[i,0]>0.5)^(test_data[i,1]>0.5)

# initialize neural network
net = nn.NeuralNetwork(layer_sizes)

# evaluate performance without training
net.print_accuracy_bool(test_data,test_labels)
net.calculate_average_cost(test_data,test_labels)

# first training session
# generate training data
training_data_1 = np.random.random((10000,2,1))
training_labels_1 = np.random.random((10000,1,1))
for i in range(0,10000):
    training_labels_1[i,0]=(training_data_1[i,0]>0.5)^(training_data_1[i,1]>0.5)

# first training session
# 5 training generations
# batch size 10 samples
# learning rate 4.0
net.train_network(training_data_1, training_labels_1, 4, 10, 4.0)

# evaluate performance after first training session
# generate test data
test_data_mid = np.random.random((1000,2,1))
test_labels_mid = np.random.random((1000,1,1))
for i in range(0,1000):
    test_labels_mid[i,0]=(test_data_mid[i,0]>0.5)^(test_data_mid[i,1]>0.5)

net.print_accuracy_bool(test_data_mid,test_labels_mid)
net.calculate_average_cost(test_data_mid,test_labels_mid)


# second training session
# generate training data
training_data_2 = np.random.random((20000,2,1))
training_labels_2 = np.random.random((20000,1,1))
for i in range(0,20000):
    training_labels_2[i,0]=(training_data_2[i,0]>0.5)^(training_data_2[i,1]>0.5)

# 8 training generations
# batch size 2000 samples
# learning rate 2.0
net.train_network(training_data_2, training_labels_2, 8, 20, 2.0)

test_data_final = np.random.random((1000,2,1))
test_labels_final = np.random.random((1000,1,1))
for i in range(0,1000):
    test_labels_final[i,0]=(test_data_final[i,0]>0.5)^(test_data_final[i,1]>0.5)

# evaluate performance after second training session
net.print_accuracy_bool(test_data_final,test_labels_final)
net.calculate_average_cost(test_data_final,test_labels_final)

# plot learned function
# this bit is very messy because I have supremely confused myself

# generate set of points across the whole space

mapdata = np.zeros(training_data_1.shape)

k=0
for i in range(0,100,1):
    for j in range(0,100,1):
        mapdata[k][0][0]=i/100
        mapdata[k][1][0]=j/100
        k+=1

# guess learned function outputs for each of the points
guess = net.predict_test(mapdata)

# create array to hold the data to be plotted
plotdat = np.zeros(training_labels_1.shape)

# flatten out the output list and put it in an array
for q in range(len(guess)):
    plotdat[q][0][0] = guess[q][0][0]

plt.imshow(plotdat.reshape(100,100),cmap = 'seismic')
plt.show()
        
    
