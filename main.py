import neuron

# Array of tuples used for training neural network
# Format is (Input1, Input2, TargetOutput)
training_data = [
                (0, 0, 0),
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
                (1, 1, 1)
]

# Array of tuples used for testing neural network
# Format is (Input1, Input2)
testing_data = [
               (0, 0),
               (0, 1),
               (1, 0),
               (1, 1)
]


def main():


    a = neuron.Neuron(1, 2)
    a.printWeight()

    # Begin Initial Testing
    print("***Begin Testing***")
    a.testNetwork(testing_data)

    # Begin Training of Netork
    print("***Begin Training***")
    a.trainNewtwork(training_data)

    # Final Testing
    print("***Begin Testing***")
    a.testNetwork(testing_data)

main()