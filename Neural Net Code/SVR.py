from sklearn.svm import SVR
from CSVReaderSpread import retrieveData
from CSVReader import oneHot
import numpy as np
from DataStandardization import standardizeData
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

'''
Use a Support Vector Machine to perform regression on spreads to predict future ones
'''

#Function to check accuracy of model
def accuracyCheck(yHats,y):
    numCorrect = 0
    if len(yHats) != len(y):
        print("ERROR: output classes of different lengths")
        return None
    totalObservations = len(yHats)
    for i in range(len(yHats)):
        if yHats[i] == y[i] or abs(yHats[i]-y[i])<3:
            numCorrect += 1
    print("Total Correct: ", numCorrect)
    print("Total Observations: ", totalObservations)
    return (numCorrect/totalObservations)*100

#Data Transformations                 
X,y,t = retrieveData("WeeklyNFLDataSPREADS.csv")
teamA = oneHot(t)

float_formatter = lambda X: "%.2f" % X
np.set_printoptions(formatter={'float_kind':float_formatter})

#print(X[:3])
#print("y: ", y[:3])
#print(teamA[:3])


#Standardize data & split into test & training data
X = standardizeData(X)
#print(X)

#combine team ID dummy var with continuous data
fullX = []
for i in range(534):
    currDataRow = X[i]
    currTeamRow = teamA[i]
    dub = np.concatenate((currTeamRow,currDataRow))
    fullX.append(dub)

fullX = np.vstack((fullX))
#print("FullX & shape: ", fullX, fullX.shape)

#Perform Cross Validation
#split data into test/train
#X_train, X_test, y_train, y_test = train_test_split(fullX, y, test_size=0.2, random_state=0)

#print("yTest: ", yTest)
print("Done with data transformations.")

'''
Begin search for best SVR model
'''
'''
print("Begin Support Vector Regression Grid Search of Kernels, C, Gamma & Epsilon parameters")

#Create pipeline with SVR
pipeline = Pipeline([('scl', StandardScaler()),
                    ('svr', SVR())])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
kernel_range = ['rbf','linear']
param_grid = [{'svr__C': param_range,'svr__gamma': param_range,'svr__epsilon': param_range,'svr__kernel': kernel_range,}]
gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='r2', cv=10, verbose=1, error_score='raise', n_jobs=4)
gs = gs.fit(fullX, y)
print("Best GridSearch Score: ", gs.best_score_)
print("Best GridSearchParameters: ", gs.best_params_)
'''


print("Begin Neural Net Regression Grid Search of solver, activation function, hidden layer size, learning rate, alpha, momentum & max iteration parameters")

#Create pipeline with mlp regressor
pipeline = Pipeline([('scl', StandardScaler()),
                    ('mlp', MLPRegressor(shuffle=True, random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
hidden_size_range = [1,5,10,25,50,100]
solver_range = ['lbfgs', 'adam']
activation_range = ['identity', 'logistic', 'tanh', 'relu']
momentum_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
max_iter_range = [25,50,100,200,300,400]
param_grid = [{'mlp__solver': solver_range,'mlp__activation': activation_range,'mlp__hidden_layer_sizes': hidden_size_range,'mlp__alpha': param_range,'mlp__learning_rate_init': param_range, 'mlp__momentum': momentum_range, 'mlp__max_iter': max_iter_range}]
gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='r2', cv=7, verbose=2, error_score='raise', n_jobs=1)
gs = gs.fit(fullX, y)
print("Best GridSearch Score: ", gs.best_score_)
print("Best GridSearchParameters: ", gs.best_params_)




'''
After finding best model, analyze it
'''

'''
#Predict spreads of test weeks based on best performing (using r^2 score) param combination form GS
yHat = gs.predict(X_test)
#Round predictions to nearest whole number
yHatRound = [ int(round(elem)) for elem in yHat ]

#check accuracy
percentCorrect = accuracyCheck(yHatRound,y_test)
print(percentCorrect,"%")


#Check for predicted vs test spreads
splits = []
i = 0 #counter for index of other array
for hat in yHatRound:
    diff = hat-y_test[i]
    splits.append(diff)
    i+=1
    
print(splits, len(splits)) 

plt.hist(splits)
plt.title("Spread Differences")
plt.xlabel("Difference Between Predicted & Actual Spreads")
plt.ylabel("Frequency")

plt.show()

'''
'''
lw=2
plt.scatter(yHatRound, y_test, color='darkorange', label='data')
plt.plot(y_test, yHatRound, color='navy', lw=lw, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
'''