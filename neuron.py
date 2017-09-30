import numpy

"""
Neuron class is a simple, single neuron, neural network.
"""
class Neuron:


    # initialize neuron with random weights
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

        # Iterate through training data
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

            # If output is too low add .25 of input to weights
            if output < target:
                self.weight += .25*inputdata

            # If output is too high subtract .25 of input to weights
            if output > target:
                self.weight -= .25*inputdata

            print(self.weight)

    # Tests neural network using testing data passed in a list of tuples
    def testNetwork(self, testingdata):

        # Iterate through testing data
        for x in testingdata:
            input1 = x[0]
            input2 = x[1]

            # Create input array using numpy
            inputdata = numpy.array([input1, input2]).T

            # Calculate net inputs using numpy dot method and pass to activation function
            netInputs = numpy.dot(self.weight, inputdata)
            output = self.activationFunction(netInputs[0])

            print("Input: " + str(input1) + " " + str(input2) + " Output: " + str(output))

    # Activation Function uses hard limit to convert input to output.
    def activationFunction(self, netInput):
        if netInput >= .75:
            return 1
        else:
            return 0

