import numpy as np
from time import time
from sklearn import preprocessing
from scipy import optimize
from CSVReader import retrieveData
from CSVReader import oneHot
from DataStandardization import standardizeData
import matplotlib.pyplot as plt
from DataStandardization import multiCollinearity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#Class is adapted and extended upon from Sebastian Raschka's Softmax Regression example in his book "Python Machine Learning" (2016)

class SoftmaxRegression(object):

    """Softmax regression classifier.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    l2 : float
        Regularization parameter for L2 regularization.
        No regularization if l2=0.0.
    minibatches : int (default: 1)
        The number of minibatches for gradient-based optimization.
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent (SGD) online learning
        If 1 < minibatches < len(y): SGD Minibatch learning
    n_classes : int (default: None)
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.

    Attributes
    -----------
    w_ : 2d-array, shape={n_features, 1}
      Model weights after fitting.
    b_ : 1d-array, shape={1,}
      Bias unit after fitting.
    cost_ : list
        List of floats, the average cross_entropy for each epoch.

    """
    def __init__(self, eta=0.01, epochs=50,
                 l2=0.0,
                 minibatches=1,
                 n_classes=None,
                 random_seed=None):

        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed

    def _fit(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_seed=self.random_seed)
            self.cost_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                    n_batches=self.minibatches,
                    data_ary=y,
                    shuffle=True):
                # givens:
                # w_ -> n_feat x n_classes
                # b_  -> n_classes
                # net_input, softmax and diff -> n_samples x n_classes:
                net = self._net_input(X[idx], self.w_, self.b_)
                softm = self._softmax(net)
                diff = softm - y_enc[idx]
                mse = np.mean(diff, axis=0)

                # gradient -> n_features x n_classes
                grad = np.dot(X[idx].T, diff)
                
                # update in opp. direction of the cost gradient
                self.w_ -= (self.eta * grad +
                            self.eta * self.l2 * self.w_)
                self.b_ -= (self.eta * np.sum(diff, axis=0))

            # compute cost of the whole epoch
            net = self._net_input(X, self.w_, self.b_)
            softm = self._softmax(net)
            cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)
        return self

    def fit(self, X, y, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_params : bool (default: True)
            Re-initializes model parametersprior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self
    
    def _predict(self, X):
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
 
    def predict(self, X):
        """Predict targets from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        target_values : array-like, shape = [n_samples]
          Predicted target values.

        """
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        Class probabilties : array-like, shape= [n_samples, n_classes]

        """
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    def _softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return - np.sum(np.log(output) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        L2_term = self.l2 * np.sum(self.w_ ** 2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)
    
    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, random_seed=None):
        """Initialize weight coefficients."""
        if random_seed:
            np.random.seed(random_seed)
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        return b.astype(dtype), w.astype(dtype)
    
    def _one_hot(self, y, n_labels, dtype):
        """Returns a matrix where each sample in y is represented
           as a row, and each column represents the class label in
           the one-hot encoding scheme.

        Example:

            y = np.array([0, 1, 2, 3, 4, 2])
            mc = _BaseMultiClass()
            mc._one_hot(y=y, n_labels=5, dtype='float')

            np.array([[1., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 1.],
                      [0., 0., 1., 0., 0.]])

        """
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)    
    
    def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
            indices = np.arange(data_ary.shape[0])
            if shuffle:
                indices = np.random.permutation(indices)
            if n_batches > 1:
                remainder = data_ary.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1],
                                                indices[-remainder:]),
                                               axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for idx_batch in minis:
                yield idx_batch
    
    def _shuffle_arrays(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]
    
    def accuracyCheck(self,yHats,y):
        numCorrect = 0
        if len(yHats) != len(y):
            print("ERROR: output classes of different lengths")
            return None
        totalObservations = len(yHats)
        for i in range(len(yHats)):
            if yHats[i] == y[i]:
                numCorrect += 1
        print("Total Correct: ", numCorrect)
        print("Total Observations: ", totalObservations)
        return (numCorrect/totalObservations)*100

# Loading Data                  
X,y,t = retrieveData("WeeklyNFLDataINJURIES.csv")
teamA = oneHot(t)

float_formatter = lambda X: "%.5f" % X
np.set_printoptions(formatter={'float_kind':float_formatter})

#Send continuous data to a multicollinearity test (doesn't return anything, just prints MC results)
#multiCollinearity(X)

#Standardize data & split into test & training data
X = standardizeData(X)


#merge team one hots with data array
#print(X.shape)
#print(teamA.shape)
fullX = []
for i in range(534):
    currDataRow = X[i]
    currTeamRow = teamA[i]
    dub = np.concatenate((currTeamRow,currDataRow))
    fullX.append(dub)
fullX = np.vstack((fullX))



#split into test/train data
#train through week 14 then test on 15-21 ###use to compare impact of injury data and non injury data performance, not for general extrapolation of data###
xTest = fullX[417:534]
xTrain = fullX[0:417]

yTest = y[417:534]
yTrain = y[0:417]



print("Done with data transformations.")


#stochastic gradient descent
nn = SoftmaxRegression(eta=0.015, epochs=494, minibatches=len(y_train), random_seed=0)


'''
batch gradient descent
nn = SoftmaxRegression(eta=0.015, epochs=300, minibatches=1, random_seed=0)
'''



nn.fit(xTrain, yTrain) #Learns model from training data xTrain and yTrain
#print(nn.w_)
#print("Costs and length: ",lr.cost_, len(lr.cost_))


#iterate over cost array to determine lowest cost at a given iteration after model fitting
lowestCost = 100
lowestIndex = 0
#parse entire cost_ list that holds avg cost for each epoch
#parse list instead of sort, b/c parse is O(n) and even mergesort is O(nlgn), and only marginal time taken here
for i in range(494): 
    if (nn.cost_[i] <= lowestCost):
        lowestCost = nn.cost_[i]
        lowestIndex = i
print("Lowest cost & index: ", lowestCost, lowestIndex)


x = nn.predict(xTest) #predicted targets saved in x from xTest(test data)
probs = nn.predict_proba(xTrain) #Predicts class probabilities from xTrain (training data)
percentageCorrect = nn.accuracyCheck(x,yTest) #predicts accuracy of x (y hat equivalent to y (actual target values)
print("Model resulted in accuracy of ", percentageCorrect, " %")


#Plot Results of Predictability
plt.plot(range(len(accuracyTrain)),accuracyTrain)
plt.title('Training Error')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
#plt.yscale('log')
plt.show()

