# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:06:00 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Kernel SVM
"""

### Intro to the Concept of Kernel SVM
"""
	1. Map to a higher dimension: so that we can get the data separate LINEARLY.
		- Use a mapping function that will create a new dimension (in 1D - y, in 2D - z) and a hyperplane that will divide the data linearly into separate classes.
		- Project the function back to a 2D space (in 2D -> 3D and back to 2D, the separation now will be shown as non-linear)
		- It's highly compute-intensive. So this approach is not the best
	2. Kernel Trick
		a. Gaussian RBF Kernel:
			§ K: kernel, x: point in data set, l^i: landmark (i many landmarks), ||x-l||: difference between, sigma: qty decided previously
			§ When you have a big distance from a point to the landmark, kernel is close to zero. When you get closer to landmark, the kernel converges to 1.
			§ The kernel in 3D space is the boundary of the dataset with the center as the landmark. Where everything outside the circumference of circle will be assigned zero and everything inside will assign a value higher than 0 and max 1.
			§ Sigma will determine the size of the circumference of the circle. A smaller sigma will take less points to be mapped and be included in the boundary for one classification.
			§ Add two (or more) kernel functions to work with two (or more) circular boundary, giving a non-linear boundary. 
		b. Types of Kernel Functions:
			§ Gaussian RBF, Sigmoid, Polynomial 
			§ Visualize in 3D: http://mlkernels.readthedocs.io/en/latest/kernels.html
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
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

### Predicting the Test set results
y_pred = classifier.predict(X_test)

### Making the Confusion Matrix
# matrix will contain the correct and incorrect predictions made by the model
from sklearn.metrics import confusion_matrix  # Import function
cm = confusion_matrix(y_test, y_pred)

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
plt.title('Kernel SVM (Training Set)')
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
plt.title('Kernel SVM (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()