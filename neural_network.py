import numpy
import csv
import scipy.special

"""
NeuralNetwork class receives number of inputs, number of hidden layers, and number of outputs
and constructs a list of lists of neurons to act as a neural network.
Example of basic 3 input, 1 hidden layer, 1 output network.
[
    [Neuron 1, Neuron 4, Neuron 7],
    [Neuron 2, Neuron 5, Empty],
    [Neuron 3, Neuron 6, Empty]
]
Neurons 1, 2, and 3 are in the input layer, 4, 5, and 6 in layer 2, and 7 is in the output layer
"""


class NeuralNetwork:

    # modifier to account for input and output layer. Added to hidden_layers to give total number of layers.
    OUTSIDE_LAYERS = 2
    WEIGHTS_FILE = "AAPLWeights.csv"
    TRAIN_FILE = "Training Data/AAPL.csv"
    LEARNING_RATE = .25

    # initialize neuron with random weights
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        self.neural_network_list = self.createNetworkList(num_inputs, hidden_layers)
        self.neural_network_weights = []
        self.neural_network_inputs = []
        self.neural_network_net_inputs = []
        self.neural_network_outputs = []

        self.network_output = 0
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.num_layers = hidden_layers + NeuralNetwork.OUTSIDE_LAYERS

        # This section initializes the weight matrix's for the neural network
        for col in range(self.num_layers):
            if col < self.num_layers - 1:
                #self.neural_network_weights.append(numpy.random.rand(self.num_inputs, self.num_inputs))
                self.neural_network_weights.append(numpy.zeros((self.num_inputs, self.num_inputs)))
                self.neural_network_inputs.append(numpy.zeros((self.num_inputs, 1)))
                self.neural_network_net_inputs.append(numpy.zeros((self.num_inputs, 1)))
                self.neural_network_outputs.append(numpy.zeros((self.num_inputs, 1)))
            else:
                #self.neural_network_weights.append(numpy.random.rand(self.num_outputs, self.num_outputs))
                self.neural_network_weights.append(numpy.zeros((self.num_outputs, self.num_inputs)))
                self.neural_network_inputs.append(numpy.zeros((self.num_inputs, 1)))
                self.neural_network_net_inputs.append(numpy.zeros((1, 1)))
                self.neural_network_outputs.append(numpy.zeros((1, 1)))

    # Creates a 2d list to act as the structure of the Neural Network. Neurons will be placed inside.
    def createNetworkList(self, num_inputs, hidden_layers):
        network_list = []
        for i in range(num_inputs):
            x = []
            for j in range(hidden_layers + NeuralNetwork.OUTSIDE_LAYERS):
                x.append(0)
            network_list.append(x)
        return network_list

    # Displays neural network structure
    def displayNeuralNetwork(self):
        for i in self.neural_network_list:
            print i

    # Using training data to train network
    def trainNetwork(self):
        with open(self.TRAIN_FILE, 'rb') as csvfile:
            reader_object = csv.reader(csvfile, delimiter=',', quotechar='/')
            for row in reader_object:
                x = 1
                for i in range(len(self.neural_network_inputs[0])): # Load inputs into layer 1
                    self.neural_network_inputs[0][i] = row[x]
                    x += 1
                target = row[self.num_inputs + 1]

                for layer in range(self.num_layers):

                    if layer < self.num_layers - 1:
                        # Get net inputs by matrix multiplication of weight and input matrix
                        self.neural_network_net_inputs[layer] = numpy.dot(self.neural_network_weights[layer], self.neural_network_inputs[layer])
                        for val in range(len(self.neural_network_outputs[layer])):
                            self.neural_network_outputs[layer][val] = self.activationFunction(self.neural_network_net_inputs[layer][val])
                        self.neural_network_inputs[layer + 1] = self.neural_network_outputs[layer]

                    else:
                        self.neural_network_net_inputs[layer] = numpy.dot(self.neural_network_weights[layer], self.neural_network_inputs[layer])
                        for val in range(len(self.neural_network_outputs[layer])):
                            self.neural_network_outputs[layer][val] = self.activationFunction(self.neural_network_net_inputs[layer][val])

                final_output = self.neural_network_outputs[self.num_layers - 1][0][0]
                error = float(target) - final_output
                hidden_error = numpy.dot(self.neural_network_weights[1].T, error)
                first_layer_error = numpy.dot(self.neural_network_weights[0].T, error)
                print ("Final Output: " + str(final_output))
                print ("Target: " + str(target))
                print ("Error: " + str(error))

                self.neural_network_weights[0] += self.LEARNING_RATE * numpy.dot((first_layer_error * final_output * 1 - final_output), self.neural_network_inputs[0])
                self.neural_network_weights[1] += self.LEARNING_RATE * numpy.dot((hidden_error * final_output * 1 - final_output), self.neural_network_outputs[1])
                self.neural_network_weights[2] += self.LEARNING_RATE * numpy.dot((error * final_output * 1 - final_output), self.neural_network_outputs[2])



    # Randomizes the weights
    def randomizeWeights(self):
        for x in self.neural_network_weights:
            for n in x:
                for i in range(len(n)):
                    n[i] = numpy.random.random_sample()

    # Imports the Weights from a CSV file specified as global variable
    def importWeights(self):
        weights_input = []
        with open(self.WEIGHTS_FILE, 'rb') as csvfile:
            reader_object = csv.reader(csvfile, delimiter=',', quotechar='/')
            for row in reader_object:
                weights_input.append(row[0])

        current_input = 0
        for x in self.neural_network_weights:
            for n in x:
                for i in range(len(n)):
                    n[i] = weights_input[current_input]
                    current_input += 1

    # Exports the Weights to a CSV file specified as global variable
    def exportWeights(self):
        with open(self.WEIGHTS_FILE, 'wb') as csvfile:
            writer_object = csv.writer(csvfile, delimiter=',', quotechar='/', quoting=csv.QUOTE_MINIMAL)
            for x in self.neural_network_weights:
                for n in x:
                    for i in range(len(n)):
                        writer_object.writerow([n[i]])

    # Displays network weights
    def displayWeights(self):
        x = 1
        for i in self.neural_network_weights:
            print "Weight Layer " + str(x)
            print i
            x += 1

    # Displays network inputs
    def displayInputs(self):
        x = 1
        for i in self.neural_network_inputs:
            print "Input Layer " + str(x)
            print i
            x += 1

    # Displays network net inputs
    def displayNetInputs(self):
        x = 1
        for i in self.neural_network_net_inputs:
            print "Net Input Layer " + str(x)
            print i
            x += 1

    # Displays network outputs
    def displayOutputs(self):
        x = 1
        for i in self.neural_network_outputs:
            print "Output Layer " + str(x)
            print i
            x += 1

    # Pass in net input, returns output using sigmoid function
    def activationFunction(self, net_input):
        output = scipy.special.expit(net_input)
        return output