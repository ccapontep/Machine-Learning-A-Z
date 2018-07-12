# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:59:35 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Hierarchical Clustering
"""

### Intro to the Concept of Hierarchical Clustering
"""
	• Same as K-means but different process
	• Two different types: 
		○ Agglomerative: Bottom-Up approach (will focus on this in the lectures)
		○ Divisive: Top-Bottom approach
	
	Agglomerative HC:
		Step 1: Make each data point a single-point cluster -> that forms N clusters
		Step 2: Take the two closest data points and make them one cluster -> that 
        forms N-1 clusters
				□ Can use different distances, in these examples will be using Euclidean
		Step 3: Take the two closest clusters and make them one cluster -> that 
            forms N-2 clusters
				□ Distance between two clusters, there are different options:
					® Option 1: Closest Points, Option 2: Furthest Point, 
                Option 3: Average Distance, Option 4: Distance Between Centroids
					® You have to decide depending on your problem which one to use
		Step 4: Repeat Step 3 until there is only one cluster
		FIN: Model done with one cluster.
		
		○ This model maintains a memory in the Dendogram: remembers every step that
        will go through
			§ Will graph the steps for Each Point vs Euclidean distance
			§ Horizontal line will connect each point that has the closest distance 
            and its area will be the actual distance between the points. So the 
            further away to points are, the higher the bar will be.
			§ Can set Dissimilarity of points (threshold) so that the height will 
            not be higher than a certain amount. Meaning the dissimilarity from
            each cluster cannot be higher than the threshold. So it will stop 
            combining clusters when before the threshold is met. 
				□ Amount of clusters can be looked at in the plot by looking at how 
                many lines 'the threshold line' touches the bars that exceed.
				□ To find the optimal number of clusters: look at the highest vertical
                distance in the dendogram that doesn't pass all the possible 
                extended horizontal "threshold" lines. And pick a threshold 
                within this distance (that crosses this largest distance). Read
                the number of clusters based on this threshold line. 
"""

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

### Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# linkage is the algorithm of hierarchical clustering, 'ward' is a method that 
# tries to minimize the within cluster variance (instead of sum of squared in k-means)
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Also finds 5 clusters

### Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

### Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc ==0, 1], s = 100, c = 'red', label = 'Careful') 
# plot scatter of the first cluster by calling the first one as found in y_hc
# to be 0, and specify the first column of data X which is 0 and 1 for x & y variables.
plt.scatter(X[y_hc == 1, 0], X[y_hc ==1, 1], s = 100, c = 'blue', label = 'Standard') 
plt.scatter(X[y_hc == 2, 0], X[y_hc ==2, 1], s = 100, c = 'green', label = 'Target') 
plt.scatter(X[y_hc == 3, 0], X[y_hc ==3, 1], s = 100, c = 'cyan', label = 'Careless') 
plt.scatter(X[y_hc == 4, 0], X[y_hc ==4, 1], s = 100, c = 'magenta', label = 'Sensible') 
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# After looking at the plot you can analyze and decide what to name each cluster
# and change it in the label as shown above.

# If want to later use K-Means clustering, just change the file name and the index 
# of the dataset.
# However, if doing clustering in more than 2D don't run the last code section to
# visualize the plot clustering. Later in course will learn how to reduce the 
# dimencion with PCA and then can run this section to visualize. 




























