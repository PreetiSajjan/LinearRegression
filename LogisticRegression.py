#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

##Reading text file
df = pd.read_csv('hazelnuts.txt', sep='\t', header=None) 
dataset = df.transpose()
column_names = ['sample_id', 'length', 'width',
                   'thickness', 'surface_area', 'mass',
                   'compactness', 'hardness', 'shell_top_radius',
                   'water_content', 'carbohydrate_content', 'variety'] 
dataset.columns = column_names

#changing the type of data to numeric
for column_name in column_names[:-1]:
    dataset[column_name] = pd.to_numeric(dataset[column_name])

#Dividing the dataset into independent and dependent variables
dataset = dataset.drop(columns = ['sample_id'])
X_data,Y_data = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values


##Implementing using scikit-learn

scikit_accuracy = np.zeros((10))
X_data = preprocessing.scale(X_data)
#Calling the logistic regression of scikit-learn
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
kfold_result = cross_val_score(logreg, X_data, Y_data, cv=10, scoring='accuracy')
scikit_accuracy = kfold_result*100

##Printing each of accuracies and average accuracy
print("Scikit-Learn")
for i in range(10):
    print("Accuracy for ", i+1, " iteration: ", scikit_accuracy[i])
print("\nAverage Test Accuracy for Scikit-Learn Logistic Regression: ", scikit_accuracy.mean(), '%')


##Logistic Regression

##Sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

##Regularized cost function
def Cost_Function(theta, X, Y, alpha = 0.09):
    m = len(Y)
    h = sigmoid(X.dot(theta)) # Calculating the probabilities for X with respect to theta
    reg = (alpha/(2 * m)) * np.sum(theta**2)
    cost = (1 / m) * ((-1 * Y * np.log(h)) - ((1 - Y) * np.log(1 - h))) + reg
    cost = cost.sum(axis = 0)
    return cost   
    
##Gradient function
def Gradient_Descent(theta, X, Y):
    cost = Cost_Function(theta, X, Y)
    m, n = X.shape
    theta = theta.reshape((n, 1))
    Y = Y.reshape((m, 1))
    h = sigmoid(X.dot(theta))    # Calculating the probabilities for X with respect to theta
    gradient = ((1 / m) * X.T.dot(h - Y))
    return cost, gradient

##Optimal theta 
def Logistic_Regression(X, Y, theta, alpha = 0.09, iterations = 20000):
    cost = np.zeros((iterations, X.shape[0]))
    #lopping over number of iterations to be performed
    for i in range(iterations):
        
        cost[i], descend = Gradient_Descent(theta, X, Y)
        theta -= alpha * descend  #Calculating the new theta value with the help of alpha- learning rate
        
    #Returning the best theta for prediction
    return theta.reshape(10)

##Predicting for testing data
def Predict(theta, X):
    h = sigmoid(X.dot(theta.T)) #probability for each nut
    prediction = [nuts[np.argmax(h[i, :])] for i in range(X.shape[0])]    #assigning highest probability class
    return prediction

##Calculating the accuracy score
def score(Y, pred):
    return sum(pred == Y)/len(Y)

##Running the implemented code
classification = []
lr_accuracy = np.zeros((10))
X_data = preprocessing.scale(X_data)

print("\nImplemented Logistic Regression")
##Iterating over the implemented LR for 10 times
for t in range(10):
    # Splitting the data into traning and testing where training data is 2/3 of dataset 
    # randomly with the help of "Shuffle"
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = 2/3, shuffle = True)
    
    nuts = np.unique(Y_data)
    #theta to hold value of thetas for all the three classes
    theta = np.zeros((3, 10)) 
    i = 0
    for nut in nuts:
        #set the labels in 0 and 1
        temp_Y = np.array(Y_train == nut, dtype = int)
        theta_optimal = Logistic_Regression(X_train, temp_Y, np.zeros((10,1)))
        theta[i] = theta_optimal
        i += 1
          
    #Predicting for X_test for every iteration
    pred = Predict(theta, X_test)
    
    for j in range(len(pred)):
        classification.append("%s, %s" %(pred[j], Y_test[j]))
    
    #Storing the accuracies of each iteration
    lr_accuracy[t] = score(Y_test, pred)*100
    print("Accuracy for ", t+1, " iteration: ", lr_accuracy[t])

##printing the average of accuracies
print("\nAverage Test Accuracy for Implemented Logistic Regression: ", lr_accuracy.mean(), '%')

##Writing the results to file
with open('prediction.csv', 'w') as writer:
                for line in classification:
                    writer.write(line + "\n")
                writer.close()

##Plotting the accuracies for both the implementations
plt.plot(range(10), scikit_accuracy, label = 'Scikit-Learn LR')
plt.plot(range(10), lr_accuracy, label = 'Implemented LR')
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.title('Accuracies for both the Implementations')
plt.legend()
plt.show        


# In[2]:


##Confusion matrix
matrix = confusion_matrix(Y_test, pred)
mat = sns.heatmap(matrix, annot = True, xticklabels = nuts, yticklabels = nuts)
mat.set(xlabel = "True Values", ylabel = "False Values")


# In[ ]:




