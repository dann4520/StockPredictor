import neural_network

a = neural_network.NeuralNetwork(7, 1, 1)

a.importWeights()
a.trainNetwork()


a.exportWeights()