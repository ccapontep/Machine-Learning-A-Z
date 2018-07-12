# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:22:15 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
K-Nearest Neighbors (K-NN)
"""

### Intro to the Concept of K-Nearest Neighbors (K-NN)
"""
	- Uses K-NN to determine where a new data point should be classified
	- How to do it:
		○ Step 1: Choose the number K of neighbors (most common default is K=5)
		○ Step 2: Take the K nearest neighbors of the new data point according to the Euclidean distance
			§ Euclidean distance between P1 and P2 = sqrt [ (x2 - x1)^2 + (y2 - y1)^2 ]
		○ Step 3: Among these K neighbors, count the number of data points in each category
		○ Step 4: Assign the new data point in the category where you counted the most neighbors
Your model is ready!
"""

###Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

###Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]]. values  
y = dataset.iloc[:, 4]. values  

### Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

### Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() #call new object to standardscaler
X_train = sc_X.fit_transform(X_train) #for training set need to fit & transform the data
X_test = sc_X.transform(X_test) #for test set need to only transform the data


### Fitting Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

### Predicting the Test set results
y_pred = classifier.predict(X_test)

### Making the Confusion Matrix
# matrix will contain the correct and incorrect predictions made by the model
from sklearn.metrics import confusion_matrix  # Import function
cm = confusion_matrix(y_test, y_pred) # 7 incorrect prediction. Better than Logistic R

###Visualising the Training set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# amplifying the min/max range so data would not look too compressed into [0,1] and set a resolution of
# 0.01 so that each pixel can show as a point and area of the classification
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# classifier.predict predicts which color each pixel observation point it corresponds to. If class 0,]
# it will be colored red. If class 1, it will classified green
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set ==j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
# loop to plot into the scatter all of the data points
plt.title('K-NN (Training Set)')
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
plt.title('K-NN (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()