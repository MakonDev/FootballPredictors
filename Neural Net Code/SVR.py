from sklearn.svm import SVR
from CSVReaderSpread import retrieveData
from CSVReader import oneHot
import numpy as np
from DataStandardization import standardizeData
import matplotlib.pyplot as plt

mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=5,
                           max_iter=300, shuffle=True, random_state=1,
                           activation='tanh')


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

#split into test/train data
#train through week 14 then test on 15-21
xTest = fullX[417:534]
xTrain = fullX[0:417]

yTest = y[417:534]
yTrain = y[0:417]

#print("yTest: ", yTest)
print("Done with data transformations.")

#Fit regression
mlp.fit(xTrain,yTrain)
#Predict spreads of test weeks 
yHat = mlp.predict(xTest)
#Round predictions to nearest whole number
yHatRound = [ int(round(elem)) for elem in yHat ]

print(yHat[:3])
print(yHatRound[:3])
print(yTest[:3])

#check accuracy
percentCorrect = accuracyCheck(yHatRound,yTest)
print(percentCorrect,"%")

#test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
#test_y = mlp.predict(test_x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(yHat, yTest, s=1, c='b', marker="s", label='real')
#ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
plt.show()
