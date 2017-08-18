import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def retrieveData(filename):
    dataArray = np.empty(shape=(534,12), dtype = np.float) 
    differentialArray = np.empty(shape=(534,), dtype = np.int)
    teamArray = np.empty(shape=(534,),dtype = np.int)
    
    f = open(filename, 'r')
    #for each line in file we split on each comma delineation and place into numpy array
    i=0 #counter for row placement into array
    for line in f:
        line = line.strip("\n") #strips \n from end of line
        #place all reader data into numpy empty array
        j=0 #counter for column array placement
        for word in line.split(","): #splits line into individual items by ","
            #for data supposed to be integers
            if j==0: #for categorical team data place into team array for future one hot conversion
                teamArray[i] = int(word)
            if j>0 and j<13:
                if word == "0":
                    dataArray[i][j-1] = float(0)
                else: #if a valid data piece
                    dataArray[i][j-1] = float(word)
                
            #for class data that's supposed to be letters
            if j==13:
                differentialArray[i] = float(word)
            j+=1 
        i+=1
        
    f.close()
    return dataArray, differentialArray, teamArray

def oneHot(y):
    
    #create array to return that will hold all 1D one hot arrays
    #tempList = []
    #for each team number determine the actual representative value
    curr = np.zeros((1,32))
    for i in range(534):
        team = y[i]
        #produce its corresponding 1D one hot encoding for that number
        temp = np.zeros((1,32))
        temp[0][team-1]=1
        #append current one hot encoding in list
        #tempList.append(temp)
        curr = np.concatenate((curr,temp))


    #format float printout
    #float_formatter = lambda x: "%.2f" % x
    #np.set_printoptions(formatter={'float_kind':float_formatter})
    
    curr = np.delete(curr,0,0) #delete first row of empty zeros
    
    #return that oneHotArray to overwrite the number only team array
    return curr


    #merge with dataArray so that all one array but AFTER data normalization of continuous data        
 