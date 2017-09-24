import numpy

class neuron:


    #initiliaze neuron
    def __init__(self):
        self.weight = 0

    def setWeight(self, weight):
        self.weight = weight

    def printWeight(self):
        print("Weight is: " + str(self.weight))

a = neuron()
a.printWeight()
a.setWeight(numpy.random.random_sample())
a.printWeight()
