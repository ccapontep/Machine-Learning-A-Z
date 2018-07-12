# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:45:27 2018

@author: Cecilia Aponte
Udemy - Machine Learning A-Z
Natural Language Processing
"""
### Intro to the Concept of NLP
"""
Natural Language Processing (or NLP) is applying Machine Learning models to text
and language. Teaching machines to understand what is said in spoken and written
word is the focus of Natural Language Processing. Whenever you dictate something
into your iPhone / Android device that is then converted to text, that’s an NLP
algorithm in action.

You can also use NLP on a text review to predict if the review is a good one or 
a bad one. You can use NLP on an article to predict some categories of the 
articles you are trying to segment. You can use NLP on a book to predict the 
genre of the book. And it can go further, you can use NLP to build a machine 
translator or a speech recognition system, and in that last example you use 
classification algorithms to classify language. Speaking of classification 
algorithms, most of NLP algorithms are classification models, and they include 
Logistic Regression, Naive Bayes, CART which is a model based on decision trees, 
Maximum Entropy again related to Decision Trees, Hidden Markov Models which are 
models based on Markov processes.

A very well-known model in NLP is the Bag of Words model. It is a model used to 
preprocess the texts to classify before fitting the classification algorithms on 
the observations containing the texts.

In this part, you will understand and learn how to:

Clean texts to prepare them for the Machine Learning models,
Create a Bag of Words model,
Apply Machine Learning models onto this Bag of Worlds model.
"""

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setting the working directory to my computer files
work_path = inspect.getfile(inspect.currentframe())
work_dir = os.path.realpath(os.path.dirname(work_path))
os.chdir(work_dir) # set working directory

### Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

### Cleaning the text
# Start by cleaning one element (review, book, text, etc) to see that is being done 
# correctly. After, add a for-loop to be done to all the rest of the elements.
import re
import nltk
nltk.download('stopwords') # download all words that are not significant to be later removed
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # import library to remove to stem of word and keep root

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # i = 0 when testing initially 1st element
    # remove punctuation characters, not the letters from a-z and capatials and replace 
    # by a space
    review = review.lower() # replace all capital letters to lower case
    review = review.split() # split the string to make a list of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # remove none significant words (articles, prepositions, etc.), do it in a loop
    # and review as a set so that it will run faster for longer text
    
    # add ps.stem to word to find the root of a word and remove the stem so that 
    # the main word will be kept and your list of words won't be too sparse
    review = ' '.join(review) # return the list into a string
    corpus.append(review)

### Creating the Bag of Words Model
# BOW model, create a column for each and all unique words in the review.
# the model will have for this example, 1000 rows for each review and k columns
# referring to the number k of unique words in all reviews. For each row, if
# the word appears then that column will a number based on the # times the word appears.
# This will be the sparce matrix, with 0 creating more sparcity. So want to reduce 
# sparcity by cleaning the data beforehand. 
# Based on the true result and the words that were used, the model will make a prediction
# through classification, so will use the CLass. template previously used.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # max_fea is for the max amount of words to 
# add to sparce matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

### Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

"""
### Fitting Naive Bayes to the Training set
# Common models for NLP are Naive Bayes and Random Forest Classification
# Other good models are CART (RForest), C5.0 and Maximum Entropy
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
"""

### Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

### Predicting the Test set results
y_pred = classifier.predict(X_test)

### Making the Confusion Matrix
# matrix will contain the correct and incorrect predictions made by the model
from sklearn.metrics import confusion_matrix  # Import function
cm = confusion_matrix(y_test, y_pred) # 10 wrong predictions


### Evaluating the Performance of the Model
tp, tn, fp, fn = cm[[1, 0, 0, 1], [1, 0, 1, 0]].ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = round(tp / (tp + fp), 2)
recall = round(tp / (tp + fn), 2)
F1_score = round((2 * precision * recall) / (precision + recall), 2)
print('Accuracy is: ' + repr(accuracy))
print('Precision is: ' + repr(precision))
print('Recall is: ' + repr(recall))
print('F1 Score is: ' + repr(F1_score))

""" 
Naive Bayes performance is as followed: 
Accuracy is: 0.73
Precision is: 0.68
Recall is: 0.88
F1 Score is: 0.77

Random Forest performance is as followed: 
Accuracy is: 0.72
Precision is: 0.85
Recall is: 0.55
F1 Score is: 0.67
"""








