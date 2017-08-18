from sklearn.svm import SVR
from CSVReaderSpread import retrieveData
from CSVReader import oneHot
import numpy as np
from DataStandardization import standardizeData
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

'''
Use a Support Vector Machine to perform regression on spreads to predict future ones
'''

svr = SVR(kernel='rbf', C=0.01, gamma=0.1, epsilon = 0.1)

#Function to check accuracy of model
def accuracyCheck(yHats,y):
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
X_train, X_test, y_train, y_test = train_test_split(fullX, y, test_size=0.2, random_state=0)

#print("yTest: ", yTest)
print("Done with data transformations.")

#Fit regression
svr.fit(X_train,y_train)
#Predict spreads of test weeks 
yHat = svr.predict(X_test)
#Round predictions to nearest whole number
yHatRound = [ int(round(elem)) for elem in yHat ]

print(yHat[:3])
print(yHatRound[:3])
print(y_test[:3])

#check accuracy
percentCorrect = accuracyCheck(yHatRound,y_test)
print(percentCorrect,"%")

lw=2
plt.scatter(yHatRound, y_test, color='darkorange', label='data')
plt.plot(y_test, yHatRound, color='navy', lw=lw, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()