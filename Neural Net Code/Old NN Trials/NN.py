import numpy as np
from sklearn import preprocessing
from scipy import optimize
from CSVReader import retrieveData
from DataStandardization import standardizeData


# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize data
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100



#Decide on either STOCHASTIC VS BATCH GRADIENT DESCENT

#Instantiation

class Neural_Network(object):
    def __init__(self,Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights
        #inputLayerSize x hiddenLayerSize array of weights between the input nodes and hidden nodes
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) 
        #hiddenLayerSize x outputLayerSize array of weights between the hidden nodes and the output node
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
        #Regularization Parameter
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propagate inputs through the network
        self.z2 = np.dot(X, self.W1) #Computes hidden layer activity by dot product of matrix of inputs and matrix of weights
        self.a2 = self.sigmoid(self.z2) #Applies signmoid activation function to activity of second layer before activation
        #Computes output layer activity by dot product of matrix of hidden layer activity and matrix of second set of weights
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) #Applies signmoid activation function to activity of output layer before activation
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to a scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
    
    def backpropagation(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y to determine gradient:
        self.yHat = self.forward(X)
        
        #calculates backpropagating error for second set of weights
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3)) 
           
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2 
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        #returns derivative of cost function wrt matrix of weights between inputs and hidden nodes
        #and the derivative of cost function wrt matrix of weights between hidden nodes and output nodes
        #or the gradients for the cost function wrt both sets of weights, independently
        return dJdW1, dJdW2
    
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.backpropagation(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        epsilon = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = epsilon
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*epsilon)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
        
        
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


'''
#Test run of forward propagation
print("Beginning of forward propagation test")
NN = Neural_Network()
yHat=NN.forward(X)
print(yHat)
print(y)
print("End of forward propagation test")
print ("")

#Test run of backpropagation
print("Beginning of backpropagation test")
NN = Neural_Network()
cost1 = NN.costFunction(X,y)
dJdW1, dJdW2 = NN.backpropagation(X,y)
print(dJdW1)
print(dJdW2)
print("End of backpropagation test")
print("")

#Numerical Gradient Checking
print("Beginning of Numerical Gradient Checking")
NN = Neural_Network()
numgrad = computeNumericalGradient(NN, X, y)
print(numgrad)
grad = NN.computeGradients(X,y)
print(grad)
#normal(grad-numgrad)/normal(grad+numgrad)
#Norm isn't recognized for some reason but we can still see the errors visually
print("End of NGC")
'''
NN = Neural_Network(Lambda=0.0001)
T = trainer(NN)
T.train(X,y)
#following block only works in ipython notebook not terminal
'''
plt.plot(T.J)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
'''
NN.backpropagation(X,y)