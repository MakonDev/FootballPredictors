import numpy as np
from sklearn import preprocessing
from scipy import optimize
from CSVReader import retrieveData

def standardizeData(array):
    standardizedArray = preprocessing.scale(array)
    return standardizedArray

def multiCollinearity(data):
    ###Cannot directly compute MC since don't have SER value due to categorical output, not continuous### 
    ###Use Variance Inflation Factor (VIF) instead to detect multicollinearity###
    ##VIF = 1/(1-correlation^2)
    
    #compute VIF for each pair of variables in data
    
    #for each column in dataset, compute mean and standard deviation
    #when comparing two given columns, calculate covariance and correlation between both then VIF value to determine
    
    #compute mean matrix
    Means = np.mean(data, axis=0)
            
    #compute standard deviation matrix
    stdDeviations = np.std(data, axis=0, ddof=1)
    
    
    #compute covariance matrix
    covarianceArray = np.zeros(shape=(12,12), dtype = np.float) 
    for x in range(12): #first column
        for y in range(12):
            if x==y: #don't want to compare column to itself
                continue
            summation = 0
            for row in range(534):
                summation = summation + ((data[row,x]-Means[x])*(data[row,y]-Means[y]))
            covarianceArray[x,y] = summation/533
       
    #compute correlation matrix
    correlationArray = np.zeros(shape=(12,12), dtype = np.float)
    for x in range(12): #first column
        for y in range(12):
            if x==y: #don't want to compare column to itself
                continue
            correlationArray[x,y] = (covarianceArray[x,y])/(stdDeviations[x]*stdDeviations[y])

    #calculate collinearity matrix
    VIFArray = np.zeros(shape=(12,12), dtype = np.float)
    for x in range(12): #first column
        for y in range(12):
            if x==y: #don't want to compare column to itself
                continue
            VIFArray[x,y] = 1/(1-correlationArray[x,y])
    print("These are the VIF's: ", VIFArray)
    
    #find max & min collinearities & their indices
    maxC = -1200
    minC = 1200
    minX, minY, maxX, maxY = 0,0,0,0
    for x in range(12): #first column
        for y in range(12):
            if x==y: #don't want to compare column to itself
                continue
            if VIFArray[x,y]<minC:
                minC = VIFArray[x,y]
                minX = x
                minY = y
            if VIFArray[x,y]>maxC:
                maxC = VIFArray[x,y]
                maxX = x
                maxY = y
    print("Max and Min Collinearities and their indices: ", maxC, maxX, maxY, minC, minX, minY)
    