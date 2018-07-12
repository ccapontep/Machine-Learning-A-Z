# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:16:35 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Decision Tree Classifier
"""

### Intro to the Concept of Decision Tree Classifier
"""
	• CART = Two types of decision trees - Classification Trees (categorical 
    variables) & Regression Trees (real number variables)
		○ For Classification assignment we will do Classification Trees 
		○ Type of Decision Trees: Random Forest and Gradient Boosting
	• Split the data so as to maximize the amount of points located in the 
    correct class 'leaf' (minimize information entropy)
	• If cannot finish the decision tree to the final terminal (the last node 
    will have a mix of both categories), then choose the category based on the 
    highest probability in that area (if more red than green, then that node 
    will call that terminal node red).
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
# Won't need to do feature scaling because this classifier is not based on Euclidean
# distances and want original value when creating the tree. However because our plot
# later brings the features beck to the original and with a resolution, it will plot
# the correct information. But if you want to plot the actual tree, you would have to 
# remove this feature scaling part. 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() #call new object to standardscaler
X_train = sc_X.fit_transform(X_train) #for training set need to fit & transform the data
X_test = sc_X.transform(X_test) #for test set need to only transform the data


### Fitting Decision Tree Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

### Predicting the Test set results
y_pred = classifier.predict(X_test)

### Making the Confusion Matrix
# matrix will contain the correct and incorrect predictions made by the model
from sklearn.metrics import confusion_matrix  # Import function
cm = confusion_matrix(y_test, y_pred) # 9 wrong predictions

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
plt.title('Decision Tree Classifier (Training Set)')
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
plt.title('Decision Tree Classifier (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""
Looks like the algorithm is overfitting for the dataset since it is trying to 
capture each data point by adding more seperation. There are methods to improve
the classifier so as to not do this. 
"""
