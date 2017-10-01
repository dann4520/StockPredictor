import numpy
import neuron

"""
NeuralNetwork class 
"""
class NeuralNetwork:

    # modifier to account for input and output layer. Added to hidden_layers.
    OUTSIDE_LAYERS = 2

    # initialize neuron with random weights
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        self.neural_network_list = self.createNetworkList(num_inputs, hidden_layers)

        self.network_output = 0
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.num_layers = hidden_layers + NeuralNetwork.OUTSIDE_LAYERS

        # This section fills the neural network with neurons.
        for row in range(len(self.neural_network_list)):
            for col in range(num_inputs):
                if row >= (num_outputs) and col > num_outputs:
                    pass
                else:
                    self.neural_network_list[row][col] = neuron.Neuron(self.num_layers)

    # creates a 2d list to act as the structure of the Neural Network. Neurons will be placed inside.
    def createNetworkList(self, num_inputs, hidden_layers):
        network_list = []
        for i in range(num_inputs):
            x = []
            for j in range(hidden_layers + NeuralNetwork.OUTSIDE_LAYERS):
                x.append(0)
            network_list.append(x)
        return network_list