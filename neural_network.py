import numpy

"""
NeuralNetwork class 
"""
class NeuralNetwork:

    # initialize neuron with random weights
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        self.neural_network_array = numpy.zeros((num_inputs, hidden_layers))
        self.ouput = 0