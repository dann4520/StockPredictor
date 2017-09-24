import numpy

class neuron:

    # initialize neuron
    def __init__(self, rows, columns):
        self.weight = numpy.random.rand(rows, columns)

    # Sets weight entry for row and column passed in
    def setWeight(self, weight, row, column):
        self.weight[row, column] = weight

    # Prints current Weights
    def printWeight(self):
        print("Weight is: " + str(self.weight))

    # Trains neural network using training data passed in as list of tuples
    def trainNewtwork(self, trainingData):
        for x in trainingData:
            input1 = x[0]
            input2 = x[1]
            target = x [2]

            # Create input array
            inputdata = numpy.array([input1, input2]).T

            # Calculate net inputs using numpy dot method and pass to activation function
            netInputs = numpy.dot(self.weight, inputdata)
            output = self.activationFunction(netInputs[0])

            print("Output: " + str(output) + " Target: " + str(target))

            if output < target:
                self.weight += .25*inputdata

            if output > target:
                self.weight -= .25*inputdata

            print(self.weight)

    # Activation Function accepts input and returns output
    def activationFunction(self, netInput):
        if netInput >= .75:
            return 1
        else:
            return 0

