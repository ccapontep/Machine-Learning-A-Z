# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:53:01 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Concept of Naive Bayes Classifier
"""

### Intro to the Concept of Naive Bayes Classifier
"""
	Bayes Theorem:
	
		P(A|B) =   P(B|A) * P(A)               
                ------------
			             P(B)
		
		where,  
			| means "given"
			A is a classification (such as drivers, walkers, etc.) that has certain features and you are trying to classify your new data point B
			B is the new data points with its features
			#1: P(A) is Prior Probability and is calculated 1st
			#2: P(B) is Marginal Likelihood and is calculated 2nd
			#3: P(B|A) is Likelihood and is calculated 3rd
			#4: P(A|B) is Posterior Probability and is calculated 4th and last
			
	Naïve Bayes:
		Step 1: Calculate probability of one classification (A, ex. walking) given the new data point (B, ex. X). 
			§ Decide the radius of the circle around the new data point, which will include the points that will be similar in features to the new data point
			§ #1: P(Walks) would be the # that walk / total observations
			§ #2: P(X) would be # of similar observations (inside the circle) / total observations. 
			§ #3: P(X | Walks)  would be the # of similar observations among those who walk (observations that walks found inside the circle) / total number of walkers
			§ #4: P(Walks | X), place all #'s above to the equation and solve
		Step 2: Calculate probability of the other classification (A, ex. driving) given the new data point (B, ex. X), which is 1 - P(Step 1)
		Step 3: Assign the classification of the new data point B, given the highest probability found in step 1 and 2.
	
		○ "Naïve" - independence assumption (assuming that the different features doesn't have a correlation, like age and salary). Still can use it
		○ P(X) can be dropped from the calculation because in the calculation it is calculated in both Step 1 and 2 so they do not affect the final probability. Don't have to calculate P(X) when you are only comparing the two. If you want the actual probability, you need to add it. 
			§ Therefore for this classifier where comparing can use the equation:       
			P(A|B) =   P(B|A) * P(A)     
		○ If have more than 2 classes, then have to calculate all probabilities without being able to do the trick of 
        1 - P(Step 1) to compare the probabilities. 
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


### Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

### Predicting the Test set results
y_pred = classifier.predict(X_test)

### Making the Confusion Matrix
# matrix will contain the correct and incorrect predictions made by the model
from sklearn.metrics import confusion_matrix  # Import function
cm = confusion_matrix(y_test, y_pred) # 10 wrong predictions

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
plt.title('Naive Bayes (Training Set)')
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
plt.title('Naive Bayes (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
