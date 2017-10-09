import numpy
import math

"""
Neuron class 
"""
class Neuron:

    # initialize neuron with random weights
    def __init__(self, num_weights):
        # self.weight = numpy.random.rand(num_weights, 1)
        self.net_input_value = 0
        self.ouput_value = 0

    def activationFunction(self, input):
        output = 1/ (1 + math.e**-input)

    def __str__(self):
        return "Input: " + str(self.net_input_value) + "\n" + "Ouput: " + str(self.ouput_value)