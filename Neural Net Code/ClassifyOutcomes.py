from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from CSVReader import retrieveData
from CSVReader import oneHot
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

'''
Need to have the process for predictions be twofold:
1) Train model on all available data, and after weeks pass by in 2017 season add each week to the training data
1a) try to see if we manually separate test/training sets or do random with the scikit learn like currently THEN test the data on that model
2) Predict outcomes for each upcoming week and output them in a very userfriendly, easy to read format (text file?)
'''


#Load Data
# Loading Data                  
X,y,t = retrieveData("WeeklyNFLDataINJURIES.csv")
teamA = oneHot(t)

float_formatter = lambda X: "%.5f" % X
np.set_printoptions(formatter={'float_kind':float_formatter})

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

#Perform Cross Validation
#split data into test/train
X_train, X_test, y_train, y_test = train_test_split(fullX, y, test_size=0.2, random_state=0)

'''
print("train X ", X_train[:3])
print("test X ", X_test[:3])
print("train y ", y_train[:3])
print("test y ", y_test[:3])
'''
print("Done with Data Transformations")

############################################################################

#Create pipeline with classifier
pipeline = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty="l2", random_state=1,solver ="lbfgs", multi_class = "multinomial"))])

'''
scores = cross_val_score(estimator=pipeline, X=X_train, y=y_train, cv=2)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print(scores)
pipeline.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipeline.score(X_test, y_test))
'''
'''
train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=2)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')

plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')

plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.show()

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipeline, X=X_train, y=y_train, param_name='clf__C', param_range=param_range, cv=2)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1.0])
plt.show()
'''

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range}]
gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=2)
gs = gs.fit(X_train, y_train)
print("Best GridSearch Score: ", gs.best_score_)
print("Best GridSearchParameters: ", gs.best_params_)

classifier = gs.best_estimator_
classifier.fit(X_train, y_train)
print('Test accuracy: %.3f' % classifier.score(X_test, y_test))

yHat = classifier.predict(X_test)
yProbs = classifier.predict_proba(X_test)
print(X_test[:3])
print("Probabilities: ", yProbs[:3])