import csv
import numpy as np

def retrieveData(filename):
    dataArray = np.empty(shape=(534,12), dtype = np.float) 
    classArray = np.empty(shape=(534,), dtype = np.int)
    teamArray = np.empty(shape=(534,),dtype = np.int)
    
    dataLength = dataArray.shape[1] #represents number of continuous variables as second element in shape tuple
    f = open(filename, 'r')
    #for each line in file split on each comma delineation and place into respective numpy array
    i=0 #counter for row placement into array
    for line in f:
        line = line.strip("\n") #strips \n from end of each line
        #place all reader data into numpy empty array
        j=0 #counter for column array placement
        for word in line.split(","): #splits line into individual items by ","
            #for data intended as ints or floats
            if j==(dataArray.shape[1]-dataArray.shape[1]): #team number data place into team array for future one hot conversion
                teamArray[i] = int(word)
            if j>(dataArray.shape[1]-dataArray.shape[1]) and j<(dataArray.shape[1]+1):
                if word == "0":
                    dataArray[i][j-1] = float(0)
                else: #if a valid data piece
                    dataArray[i][j-1] = float(word)
                
            #for class data that's supposed to be letters
            if j==(dataArray.shape[1]+1):
                if word=="L":
                    classArray[i] = 0
                elif word=="T":
                    classArray[i] = 1
                else: #if it equals "W"
                    classArray[i] = 2                
            j+=1 
        i+=1
        
    f.close()
    return dataArray, classArray, teamArray

def oneHot(y):
    
    #create array to return that will hold all 1D one hot arrays
    #for each team number determine the actual representative value
    oneHots = np.zeros((1,32))
    for i in range(534):
        team = y[i]
        
        #produce its corresponding 1D one hot encoding for that number
        temp = np.zeros((1,32))
        temp[0][team-1]=1
        
        #concatenate current one hot encoding in list
        oneHots = np.concatenate((oneHots,temp))
    
    oneHots = np.delete(oneHots,0,0) #delete first row of empty zeros
    
    #return that oneHotArray to overwrite the representative team number array
    return oneHots

    #merge with dataArray so that all one array but AFTER data normalization of continuous data        
 