# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:40:27 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Logistic Regression
"""

### Intro to the Concept of Logistic Regression
"""

Logistic Regression:
	Uses a Sigmoid function: p = 1 / (1 + e^-y) to get the logistic regression 
    formula that is:
	
		Ln ( p / (1 - p) = b0 + b1 * x
		
Which predicts probability (y^ : p-hat which is the predicted dependent variable)
of something happening depending where they lie on the model.

Goal of logistic regression is to classify the independent variable to the 
correct class dependent variable.
"""

###Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

###Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]]. values  # create independent variable,                           a matrix and not a vector
y = dataset.iloc[:, 4]. values  #create dependent variable

### Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split #import library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
#seperate and link the seperate training sets. 

### Feature Scaling
from sklearn.preprocessing import StandardScaler #import library
sc_X = StandardScaler() #call new object to standardscaler
X_train = sc_X.fit_transform(X_train) #for training set need to fit & transform the data
X_test = sc_X.transform(X_test) #for test set need to only transform the data



### Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

### Predicting the Test set results
y_pred = classifier.predict(X_test)

### Making the Confusion Matrix
# matrix will contain the correct and incorrect predictions made by the model
from sklearn.metrics import confusion_matrix  # Import function
cm = confusion_matrix(y_test, y_pred)
# shows that there are 65 and 24 correct predictions also 3 and 8 incorrect predictions


###Visualising the Training set results 
# red points are users that didn't buy, and green that did buy
# shows that users with lower salary and younger didn't buy; older users with high
# and some with low salary also bought.
# Prediction boundary is what divides the classification
# In this case the bounday is a straight line, so it is a linear classifier

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# amplifying the min/max range so data would not look too compressed into [0,1] and set a resolution of
# 0.01 so that each pixel can show as a point and area of the classification
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# apply the classifier to each pixel observation point; colors the points to the corresponding color
# uses contour to create that bounday line between the two colors
# classifier.predict predicts which color each pixel observation point it corresponds to. If class 0,]
# it will be colored red. If class 1, it will classified green
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
# loop to plot into the scatter all of the data points
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
                     
###Visualising the Test set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Trest Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

