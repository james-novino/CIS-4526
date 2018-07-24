#Name: James Novino
#TUID: 914968597
#Email: james.novino@temple.edu
#Course: Fundamentals of Machine Learning
#Due Date: Tuesday, October 18, 2016
#Professor: Richard Souvenir

import matplotlib.pyplot as plt
import numpy as np
import random
import operator as op
import time
import os
import sys


#Globals to define plotting
show = False #Will open the plots generated 
save = False #Will save the plots to a /Plot folder in current directory
#If both false will not plot. 

#Name: main
#Input: None
#Output: None
#Summary: Runs the PLA and pseudoinverse functions for all points in N and averages
#the results over n number of iterations.
def main ():
    n = 1000 #Number of iterations to average over
    N = [10, 50, 100, 200, 500, 1000]
    w0 = np.zeros(3)
    #Run average simultations for uninitialized PLA
    for j in range(0, len(N)): 
        average = 0; elapsed = 0.0
        for i in range(0, n):       
            [X,Y] = generateData(N[j])
            #Average the results 
            start = time.time() # Start Timer        
            [w, iters] = pla(X, Y, w0)
            end = time.time()
            elapsed += end-start
            average += iters
        total = elapsed
        average = average/n; elapsed = elapsed/n
        print('\t\t\tPerceptron Learning Algorithm (PLA) \nAverage Execution Time: %s seconds| Total Execution Time: %s seconds| N:%d | Average Iterations: %s\n' %(str(elapsed),str(total), N[j], str(average)))

    print('\n###################################################################')
    print('                             PseudoInverse                           ')    
    print('###################################################################')
    #Run average simultations for initialized PLA
    for j in range(0, len(N)):  
        average = 0; elapsed = 0.0
        for i in range(0, n):
            [X, Y] = generateData(N[j])    
                    
            w = pseudoinverse(X, Y)
            #Average the results
            start = time.time() #Start Timer
            [w, iters] = pla(X, Y, w)
            end = time.time() #
            elapsed += end-start
            average += iters
        total = elapsed
        average = average/n; elapsed = elapsed/n
        print('Perceptron Learning Algorithm (PLA) Intialized with Pseudo Inverse\nAverage Execution Time: %s seconds| Total Execution Time: %s seconds| N:%d | Average Iterations: %s\n' %(str(elapsed),str(total), N[j], str(average)))
            
###############################################################################
#                   Helper Functions                                          #
###############################################################################   
#Name: generateData
#Input: N
#Output: None
#Summary: Tests the testKnn function for all 30 cases and prints 
#results to file
def generateData(N):
    Y = np.zeros((N,1))    
    X = np.zeros((N,2))
    p = np.zeros((2,2))

    #Generate Points
    for i in range(0, N):
        X[i] = [random.uniform(-1, 1) for i in range(2)]
        
    #Select Random two points and generate line
    p[0] = X[random.randint(0, N-1)]
    p[1] = X[random.randint(0, N-1)]
    coefficients = np.polyfit(p[0], p[1], 1)

    # Compute line and classify data on either side of the line
    polynomial = np.poly1d(coefficients)
    tmp = polynomial(X[:,0])
    for i in range(len(X)):
        if(tmp[i] > X[i][1]):
            Y[i] = 1
        else: 
            Y[i] = -1
    #Create plot for traiing data and decision boundary
    plot(X, Y, p, [], 'Generate Data', show=show, save=save)
    return(X,Y)

##Name: calc_error
#Input: h(ndarray), y(ndarray)
#Output: (error, index)
#Summary: Function calculates the number of errors between h (response) and 
#y the truth values, will return the number of errors and an randomly selected
#index from the incorrectly classified points.
def calc_error(h,y):
    
    incorrect = []; error = 0; idx = 0
    for i in range(0, len(h)):
        if(h[i]!=y[i]): 
            error += 1 
            incorrect = np.append(incorrect, i)
            #print('h[%d]: %s, y[%d]: %s, val:%s, error:%d' %(i, str(np.sign(h[i])), i, str(np.sign(y[i])), str(h[i]!=y[i]), error))
    if(np.any(incorrect)): idx = np.random.choice(incorrect)
    return(error, idx )
        
    
#Name: plot
#Input: X (nd.array), Y(nd.array), p(nd.array), title(str), show(Bool), save(Bool)
#Output:None
#Summary: This function will plot the points and the decision boundary. If the show
#flag is true will display the plots, if the save flag is true will save the plots
#to an output folder in the same directory
def plot(X, Y, p=[], w=[], title="", show=False, save=False):
    if(show or save):    
        fig = plt.figure()
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)
        x_axis = np.linspace(-1,1,100)
        y_axis = np.zeros(x_axis.shape)
        if((w == [])):
            coefficients = np.polyfit(p[0], p[1], 1)
            polynomial = np.poly1d(coefficients)
            y_axis = polynomial(x_axis)
            plt.plot(x_axis, y_axis, 'b-', lw=2)
            plt.title(title)
        else:
            a, b = -w[1]/w[2], -w[0]/w[2] 
            plt.plot(x_axis, a*x_axis+b,'b-', lw=2)
    
        plt.title(title)
        for i in range(len(X)):
            if(Y[i] > 0):
                plt.plot(X[i,0], X[i,1], 'go')
            else:
                plt.plot(X[i,0], X[i,1], 'ro')
        if(save):
            # Find path and create directory path and filename
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, 'Plot/')
            print(results_dir)
            sample_file_name = '%s_N_%s' % (title, str(len(Y)))
            
            #If Plot folder doesn't exist create it 
            if not os.path.isdir(results_dir): 
                os.makedirs(results_dir)
            plt.savefig(results_dir + sample_file_name)
            
        if(show):
            plt.show()


###############################################################################
#                           Algorithms                                        #
###############################################################################

#Name: pla
#Input: X(numpy.ndarray), Y(numpy.ndarray), w0(numpy.ndarray)
#Output: w (numpy.ndarray), iters(int)
#Summary: returns a learned weight vector and number of iterations for the perceptron 
#learning algorithm. w0 is the (optional) initial set of weights.
def pla(X, Y, w0=np.zeros(3)): 
 
    #Initalized required variables
    error = len(X); h = np.zeros((Y.shape)); w = np.copy(w0)
    new_X = np.zeros((len(X), len(X[0])+1)); done = False
    incorrect = np.zeros((Y.shape))
    new_X = np.concatenate((np.ones((len(X), 1)), X), axis=1) #Add Bias
    i = 0 #Iterations
    
    while not done:
        i += 1  
        #calcualte label based on weight
        h = np.sign(np.dot(new_X, w))
        error, idx = calc_error(h,Y)  
        #If errors then update weights        
        if(error != 0):
            w += (Y[idx][0]*new_X[idx]) 
        else:
            done = True

    plot(X, Y, [], w, "PLA Intialized", show=show, save=save)
    return(w, i)

#Name: pseudoinverse
#Input: X(numpy.ndarray), Y(numpy.ndarray)
#Output: w(numpy.ndarray)
#Summary: returns the learned weight vector for the pseudoinverse algorithm for linear regression.
def pseudoinverse(X, Y):
    new_X = np.zeros((len(X), len(X[0]+1)));
    new_X = np.concatenate((np.ones((len(X), 1)), X), axis=1) #Add Bias
    #Calcualte pseudo inverse    
    inverse = (np.linalg.pinv(new_X)*new_X.T) #calculate pseudo inverse    
    w = np.dot(inverse,Y)
    return(w.T[0])

if __name__ == "__main__":
    sys.exit(main())
