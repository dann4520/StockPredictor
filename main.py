import SingleNeuron

# Array of tuples used for training neural network
trainingData = [(0, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (1, 1, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (1, 1, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (1, 1, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 0),
                (1, 1, 1)]

# Array of tuples used for testing neural network
testingData = [(0, 0),
               (0, 1),
               (1, 0),
               (1, 1)]

def main():

    a = SingleNeuron.neuron(1, 2)
    a.printWeight()
    a.trainNewtwork(trainingData)
    a.printWeight()
main()