# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:36:04 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
K-Means Clustering
"""

### Intro to the Concept of K-Means Clustering
"""
	Step 1: Choose the number of K clusters
	Step 2: Select at random K points, the centroids (not necessarily from your dataset)
	Step 3: Assign each data point to the closest centroid -> that forms K clusters. 
		Quicker method is to find the line that is equidistant to the centroids and assign all points to one side to that cluster and the other side for the other cluster.
		Can use any type of distance measurement, you decide which one to use.
	Step 4: Compute and place the new centroid of each cluster
		Find the center of gravity for all the data points of the cluster and move it to that location.
	Step 5: Reassign each data point to the new closest centroid. If any reassignment took place, go to Step 4, otherwise go to FIN.
	FIN: Your model is ready!
	
	Random Initialization Trap:
		○ Have a situation that the centroids are picked incorrectly and creates a different and wrong clustering than what the data set should be. 
		○ The K-Means algorithm chooses at random the starting centroids, so to solve this we need to do a modification of the algorithm to correctly choose the starting centroids. Use K-Means++.
		○ Make sure the tools that you use are for correcting this.
	
	Selecting the Number of Clusters:
		○ "Within Cluster Sum of Squares": is a metric that can be used to impose the clustering algorithm that will tell us something about the result. 
			§ Summing every point in cluster 1 of the distance between every point and it's cluster centroid to the power of two, plus the same for cluster 2, plus the same for cluster 3, and etc. until n clusters.
			§ Start by using one cluster, then increase to two and get the WCSS. 
			§ Then add another cluster and get the WCSS.
			§ Compare the WCSS of the previous # of clusters to the current one to decide how many clusters to keep or to keep adding centroids. 
				□ As you add clusters, WCSS will decrease. Plot WCSS vs the number of clusters to see how it changes.
				□ Use elbow method to see where the drop goes from being substantial to not as great. And this will become the number of clusters to keep.
				□ Sometimes the elbow won't be too obvious, and in that case you will have to make a judgment call to decide about what to choose based on what you have. 
"""

### Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

### Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    # Fit K-means algorithm to data X. i is the number of cluseters which we are 
    # testing. init is the initialization and we don't want to do random so we 
    # use the k-means++, the maximum amount of iterations that the algorithm will
    # run is 300 which is the default, n_init is the number of times the algorithm
    # will run with different initial centroids
    wcss.append(kmeans.inertia_)
    # compute the wcss and append it to wcss list
    # another name of wcss is inner shell. Intertia_ is the name of computing wcss

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Elbow is present at 5 clusters. So need only 5 clusters

### Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X) 

### Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans ==0, 1], s = 100, c = 'red', label = 'Careful') 
# plot scatter of the first cluster by calling the first one as found in y_kmeans
# to be 0, and specify the first column of data X which is 0 and 1 for x & y variables.
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans ==1, 1], s = 100, c = 'blue', label = 'Standard') 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans ==2, 1], s = 100, c = 'green', label = 'Target') 
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans ==3, 1], s = 100, c = 'cyan', label = 'Careless') 
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans ==4, 1], s = 100, c = 'magenta', label = 'Sensible') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
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
# dimension with PCA and then can run this section to visualize. 























