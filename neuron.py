import numpy

"""
Neuron class 
"""
class Neuron:

    # initialize neuron with random weights
    def __init__(self, num_weights):
        self.weight = numpy.random.rand(num_weights, 1)
        self.input = 0
        self.ouput = 0