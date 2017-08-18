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
            print(len(outputs), "Outputs")
            print("Expected:", expected)
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backwardPropagation(network, expected)
            update_weights(network, row, learningRate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sum_error))
        
        
# Make a prediction with a network
def predict(network, row):
    outputs = forwardPropagate(network, row)
    #return outputs.index(outputs) #max takes largest probability and picks it we want the probability itself in real case
    return outputs
    
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
n_inputs = len(dataset[0]) - 1
n_outputs = 2
network = initializeNeuralNet(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
print("Layers below:")
for layer in network:
    print(layer)
for row in dataset:
    prediction = predict(network, row)
    print("Likelihood of correct prediction:", prediction)