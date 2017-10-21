import neural_network
import numpy
import math

a = neural_network.NeuralNetwork(7, 1, 1)

a.neural_network_inputs[0][0] = .76
a.neural_network_inputs[0][1] = .1
a.neural_network_inputs[0][2] = .88
#a.neural_network_inputs[0][3] = .76
#a.neural_network_inputs[0][4] = 2
#a.neural_network_inputs[0][5] = -5
#a.neural_network_inputs[0][6] = -5

x = 1
for i in a.neural_network_weights:
    print "Weight Layer " + str(x)
    print i
    x += 1

x = 1
for i in a.neural_network_inputs:
    print "Input Layer " + str(x)
    print i
    x += 1

x = 1
for i in a.neural_network_net_inputs:
    print "Net Input Layer " + str(x)
    print i
    x += 1

x = 1
for i in a.neural_network_outputs:
    print "Output Layer " + str(x)
    print i
    x += 1

a.importWeights()
a.neural_network_net_inputs[0] = numpy.dot(a.neural_network_weights[0], a.neural_network_inputs[0])

x = 1
for i in a.neural_network_net_inputs:
    print "Net Input Layer " + str(x)
    print i
    x += 1

x = 1
for i in a.neural_network_outputs:
    print "Output Layer " + str(x)
    print i
    x += 1

for x in range(len(a.neural_network_outputs[0])):
    a.neural_network_outputs[0][x] = 1/ (1 + math.e**-a.neural_network_net_inputs[0][x])

# [0] = 1/ (1 + math.e**-a.neural_network_outputs[0][0])

x = 1
for i in a.neural_network_net_inputs:
    print "Net Input Layer " + str(x)
    print i
    x += 1

x = 1
for i in a.neural_network_outputs:
    print "Output Layer " + str(x)
    print i
    x += 1

x = 1
for i in a.neural_network_weights:
    print "Weight Layer " + str(x)
    print i
    x += 1

a.randomizeWeights()

x = 1
for i in a.neural_network_weights:
    print "Weight Layer " + str(x)
    print i
    x += 1


a.exportWeights()