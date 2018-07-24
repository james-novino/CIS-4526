# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:36:43 2016

@author: james
"""
#This one is very slow :(
import csv
import numpy as np
import time
import collections
import sys
import random
import scipy.spatial as scipy

#Set Condessed Flag 
condense = True #If one will run condesing algorithm
#Set Confusion Flag
confusion = False #If true will print confusion matrix

#Name: main
#Input: None
#Output: None
#Summary: Tests the testKnn function for all 30 cases and prints 
#results to file
def main():
    #sys.stdout = open('./result.txt', 'w')
    
    #Initalize k-num indexs for algorithm
    k = [1,3,5,7,9]
    N = [100, 1000, 2000, 5000, 10000, 15000]
    #k = [1]
    #N = [100]
    #File Path for Training Data
    training_file = './lettertraining.txt'
    testing_file = './lettertest.txt'

    
    for i in range(0, len(k)):
        for j in range(0, len(N)):
            print("Condensed k-NN classifier")
            print("N: ", N[j], " k: ", k[i]) 
            #load & Format Training Data
            trainX = load(training_file)
            #Load & Format Testing Data
            testX = load(testing_file)
            #Randomly Sample Data to N Points    
            trainX = sample(trainX, N[j])  
            #Copy Training Set into Label Answer Set
            trainY = np.copy(trainX[:,0], order='F')
            trainY = trainY.astype(str, copy=False, order='F')
            #Delete Label Column from Training Set
            trainX = np.delete(trainX, 0, 1)
            trainX = trainX.astype(int, copy=False, order='F') #Convert from character to int type  
            if(condense):
                start = time.time() # Start Timer
                index = condensedata(trainX, trainY) 
                print("n(condensed): ", len(index))
                trainX = rebuild(index, trainX)  
                trainY = rebuild(index, trainY)
                end = time.time()
                elapsed = end - start   
                print('condensedata: ', elapsed, ' s ' ) 
            #Copy Training Set into Label Answer Set
            testYCorrect = np.copy(testX[:,0])
            testX = np.delete(testX, 0, 1)
            testX = testX.astype(int, copy=False, order='F') #Convert from character to int type  
            
            #Algorithmn Function Calls
            start = time.time() # Start Timer
            testY = testknn(trainX, trainY, testX, k[i])
            error(testY, testYCorrect)
            end = time.time()
            elapsed = end - start   
            print('testknn: ', elapsed, ' s ' ) 
            print()
            
#Name: condensedata
#Input: trainX(numpy array), trainY (numpy array)
#Output: condensedIdx (numpy array)
#Summary: Function will parse trainX and return the minimum number of points
#Needed to maintain preformance of the testknn algorithmn
def condensedata(trainX, trainY):
    index = np.empty(0,int)
    new_trainX = np.zeros_like(trainX)
    new_trainY = np.zeros_like(trainY)
    new_trainX = np.delete(new_trainX, range(1, len(new_trainX)), 0)
    new_trainY = np.delete(new_trainY, range(1, len(new_trainY)))
    random_index = random.randint(0,len(trainX))
    new_trainX[0] = trainX[random_index,:]
    new_trainY[0] = trainY[random_index]
    index = np.concatenate( (index, [random_index]), axis=0)
    #index = np.append(index, [random_index], axis=0)
    flag = 1
    while(flag == 1):
        
        testY = testknn(new_trainX, new_trainY, trainX, 1)

        #Reset incorrect counter
        incorrect = 0
        incorrect = (testY != trainY).sum() #count number of incorrect
        error_index = np.array(incorrect,int)
        error_index = np.where(testY!=trainY)[0] #Find all the error indexs
       
        if(incorrect == 0): flag = 0  
#        #elif(incorrect > math.ceil(len(testY)/2)):
#            for j in range(0, math.ceil(incorrect/2)):
#                random_index = random.choice(error_index)
#                new_trainY = np.concatenate( (new_trainY, [trainY[random_index]]), axis=0 )
#                #new_trainY = np.append(new_trainY, [trainY[random_index]], axis=0)
#                #new_trainX = np.append(new_trainX, [trainX[random_index]], axis=0)
#                new_trainX = np.concatenate( (new_trainX, [trainX[random_index]]), axis=0 )
#                index = np.concatenate( (index, [random_index]), axis=0)
        else:
            random_index = random.choice(error_index)
            new_trainY = np.concatenate( (new_trainY, [trainY[random_index]]), axis=0 )
            new_trainX = np.concatenate( (new_trainX, [trainX[random_index]]), axis=0 )
            #index = np.append(index, [random_index], axis=0)
            index = np.concatenate( (index, [random_index]), axis=0)
    
    index = np.sort(index)
    return(index)    
        
#Name: rebuild
#Input: index(numpy.ndarray), array(numpy.ndarray)
#Output: new_array(numpy.ndarray)
#Summary: This will rebuild the training set with the corresponding indexs
def rebuild(index, array):
    new_array = np.empty_like(array)
    new_array = np.delete(new_array, range(1,len(new_array)), 0)
    new_array[0] = array[0]
    for i in range(1, len(index)):
        new_array = np.append(new_array, [array[i]], axis=0)
    return(new_array)
    
#Name: sample
#Input: arr(Array), N(int)
#Output: mask(downsampled array)
#Summary: This will down sample the input array to contain N points
#referenced: http://docs.scipy.org/doc/numpy-1.10.0/reference/arrays.indexing.html
def sample(arr, N):
    return (arr[::round(len(arr)/N),:])

#Name: convert_char
#Input: old(char)
#Output: (int)
#Summary: Function will convert character into an index value 0-25
def convert_char(old):
    if len(old) != 1:
        return 0
    new = ord(old)
    if 65 <= new <= 90:
        # Upper case letter
        return new - 65
    elif 97 <= new <= 122:
        # Lower case letter
        return new - 97
    # Unrecognized character
    return 0
    
#Name: error
#Input: testY(numpy.ndarray), testYCorrect(numpy.ndarray)
#Output: correct(int)
#Summary:Will calculate the preformance of the predictions vs. the 
#truth values will return the number of correct. If confusion matrix flag is
#enabled will generate a confusion matrix as well
def error(testY, testYCorrect):
    correct = 0
    incorrect = 0
    if(confusion):
        result_file = './confusion.csv'
        confusion_matrix = np.zeros((26,26), dtype=int)
        correct = (testY == testYCorrect).sum()
        incorrect = (testY == testYCorrect).sum()
        for i in range(len(testY)):
            confusion_matrix[convert_char(testY[i])][convert_char(testYCorrect[i])]+=1
        np.savetxt(result_file, confusion_matrix, delimiter=',', fmt='%u')#Print matrix to file
    else:
        correct = (testY == testYCorrect).sum()
        incorrect = (testY == testYCorrect).sum()
    print("Correct: ", correct, " Incorrect: ", incorrect, " Total: ", len(testY))
    print("Accuracy: ",(correct/len(testY))*100, "%")
    return(correct)
    
#Name: load
#Input: filename(string)
#Output: data(numpy.ndarray)
#Summary: Function loads data from CSV file and formats data into
#numpy array
def load(filename):
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',', 
                               quotechar = '"')
        data = [data for data in data_iter]
    
    return np.asarray(data)

#Name: testknn
#Input: trainX (nTrain*D), trainY(nTrain*1), testX(nTest*D)
#Output: testY (nTrain*1)
#Summary: Implementation of K-NN Machine Learning Algorithmn
#Calculates euclidean distance between points to determine label
#label are found in both trainY and testY
def testknn(trainX, trainY, testX, k):
    
    testY = np.zeros(len(testX), dtype=str) # Initalize TestY to Zero
    tmpY = np.copy(trainY, order='F')

    for i in range(0, len(tmpY)):
        tmpY[i] = ord(trainY[i])

    for i in range(0, len(testX)):  
        trainDst = knn(trainX, testX[i], len(trainY))              
        #Concatenate Distances onto Label Array
        #Sort take the k number of distances and find the maximum number of labels   
        dstIndex = np.argsort(trainDst)
        
        #Minor improvement in runtime
        if k == 1:                      
            testY[i] = trainY[dstIndex[0]]
        else:
            counter = np.zeros(k, dtype=str)  
            string = ""  
            for j in range(k):
                counter[j] = trainY[dstIndex[j]]
                
            string = string.join(counter)
            testY[i] = collections.Counter(string).most_common(1)[0][0]
                
    return testY

#Name: knn
#Input: trainX (), test(), size (int)
#Output: dst(float)
#Summary: function calculates the distance for every point in 
#trainX from the test point (test)
def knn(trainX, test, size):
    
    dst = np.zeros(size, dtype=np.float, order='F') #Set Float Type
    new_test = np.array([test,]*1)
    #Calculate distance from testpoint on every point in training set
    dst = scipy.distance.cdist(trainX, new_test, 'euclidean')
    return(dst.reshape((len(dst)),))


main()