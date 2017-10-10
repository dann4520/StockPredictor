import numpy
import neuron

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

        # This section fills the neural network with neurons.
        # Number of rows depends of number of inputs.
        # Number of columns depends on number of layers.
        for row in range(self.num_inputs):
            for col in range(self.num_layers):
                if row >= num_outputs and col > num_outputs:
                    pass # If in last row only create enough neurons to account for outputs
                else:
                    self.neural_network_list[row][col] = neuron.Neuron(self.num_inputs)

        # This section initializes the weight matrix's for the neural network
        for col in range(self.num_layers):
            if col < self.num_layers - 1:
                self.neural_network_weights.append(numpy.random.rand(self.num_inputs, self.num_layers))
                self.neural_network_inputs.append(numpy.zeros((self.num_inputs, 1)))
                self.neural_network_net_inputs.append(numpy.zeros((self.num_inputs, 1)))
                self.neural_network_outputs.append(numpy.zeros((self.num_inputs, 1)))
            else:
                self.neural_network_weights.append(numpy.random.rand(self.num_outputs, self.num_layers))
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
        pass



