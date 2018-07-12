# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 18:30:38 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Apriori
"""

### Intro to the Concept of Apriori
"""
	• Things in similarities (ex. People who bought x also bought y) given prior information
	• Basic way to do this task, not as complex as those used in Netflix, Amazon, etc. 
	Support:
			§ Example: for a movie recommendation
			Support (M) = # user watchlists containing M / # user watchlists
			§ Same as Naïve Bayes
	Confidence:
			§ Hypothesis that you are trying to test to see if there are similarities
			§ Example: for a movie recommendation
			Confidence (M1 -> M2) = # user watchlists containing M1 and M2 / # user watchlists containing M1
	Lift:
			§ Likelihood of picking the second variable given the first
			§ Lift = Confidence / Support
			§ Example: movie recommendation
			Lift (M1 -> M2) = 
	
	• Algorithm Step sequence:
		Step 1: Set a minimum support and confidence
				□ Give a minimum support and confidence to limit items that have a low success rate. 
				□ Otherwise the algorithm will be too big since there can be so many connections and will run too slow
		Step 2: Take all the subsets in transactions having higher support than minimum support 
		Step 3: Take all the rules of these subsets having higher confidence than minimum confidence
		Step 4: Sort the rules by decreasing lift
"""
### Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

### Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

### Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
"""
min_support, min_confidence, min_lift can be decided by your data and can be changed
later if you don't see much result from the association.
In this case will use min_support = 0.003, because choosing a mimimum of an item being
bought 3 times a day which in a week is equal to 7 and divided by 7500 total purchases.
min_lift = 0.2; started with 0.8 (80%) but were getting too many obvious rules because
they are common bought items not because of a similarity to the rule. So halved it and 
still got same problem, so halved it again and got good rules with this value.
min_lift = 3 to get relevant rules
min_length = 2 because want to find similarities when buying more than one item,
otherwise can't compare.
"""


















