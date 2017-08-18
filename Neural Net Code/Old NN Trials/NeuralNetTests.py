from random import seed
from random import random
from math import exp

#Initialize the Network
def initializeNeuralNet(nInputs, nHidden, nOutputs):
    network = list()
    hiddenLayer = [{'weights':[random() for i in range(nInputs + 1)]} for i in range(nHidden)]
    network.append(hiddenLayer)
    outputLayer = [{'weights':[random() for i in range(nHidden + 1)]} for i in range(nOutputs)]
    network.append(outputLayer)
    return network

#Calculate neuron activation for an input, or the output from a given layer
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        #Sum of all weights*their respective inputs
        activation += weights[i] * inputs[i] 
    return activation #weighted output of a given layer

#Transfer neuron activation
def transfer(activation):
    return 1.0/(1.0 + exp(-activation)) #sigmoid activation

#Forward propagate input to a network output
def forwardPropagate(network, row):
    inputs = row
    for layer in network:
        newInputs = []
        for neuron in layer:
            activation  = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            newInputs.append(neuron['output'])
        inputs = newInputs
    return inputs

#Calculate derivative of sigmoid transfer function to find its slope
def signmoidDerivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backwardPropagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * signmoidDerivative(neuron['output'])

#Update network weights with error
def update_weights(network, row, learningRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learningRate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, learningRate, epochs, outputs):
    for epoch in range(epochs):
        sum_error = 0
        for row in train:
            outputs = forwardPropagate(network, row)
            expected = [0 for i in range(len(outputs))]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backwardPropagation(network, expected)
            update_weights(network, row, learningRate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sum_error))
        
        
# Make a prediction with a network
def predict(network, row):
    outputs = forwardPropagate(network, row)
    return outputs.index(max(outputs)) #max takes largest probability and picks it we want the probability itself in real case
        
#Test Initialization
#seed(1)
#network = initializeNeuralNet(2, 1, 2)
#for layer in network:
#    print(layer)
    
#Test forward propagation
#network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#network = initializeNeuralNet(2,1,2)
#row = [1,0]
#row = [1,0, None]
#output = forwardPropagate(network,row)
#print(output)

# test Backpropagation of error
#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#        [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
#expected = [0, 1]
#backwardPropagation(network, expected)
#for layer in network:
#    print(layer)

# Test training backprop algorithm
#seed(1)
#dataset = [[2.7810836,2.550537003,0],
#    [1.465489372,2.362125076,0],
 #   [3.396561688,4.400293529,0],
#    [1.38807019,1.850220317,0],
#    [3.06407232,3.005305973,0],
#    [7.627531214,2.759262235,1],
#    [5.332441248,2.088626775,1],
#    [6.922596716,1.77106367,1],
#    [8.675418651,-0.242068655,1],
#    [7.673756466,3.508563011,1]]
#n_inputs = len(dataset[0]) - 1
#n_outputs = len(set([row[-1] for row in dataset]))
#network = initializeNeuralNet(n_inputs, 2, n_outputs)
#train_network(network, dataset, 0.5, 20, n_outputs)
#for layer in network:
#    print(layer)

# Test making predictions with the network
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
    [{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))
    
