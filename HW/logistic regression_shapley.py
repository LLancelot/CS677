# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:12:39 2019

@author: epinsky
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

url = r'https://archive.ics.uci.edu/ml/'  + \
           r'machine-learning-databases/iris/iris.data'

data = pd.read_csv(url, names=['sepal-length', 'sepal-width', 
                     'petal-length', 'petal-width', 'Class'])

# consider 3 different grouping of classes (one against all)
# (1) Iris setosa vs. Versicolor + virginica
# (2) Versicolor vs. setosa + virginica
# (3) virginica vs. versicolor + setosa

features = ['sepal-length', 'sepal-width','petal-length', 'petal-width']

# for each grouping above, you run logistic regression on all 4 features 
# compute accuracy A
# remove one feature at a time, run logistic regression and compute
# accuracy A1, A2, A3, A4
# compute marginal contributions to accuracy
# d1, d2, d3, d4


# need to modify code below for these groupings ......

# IMPORTANT: NEED TO SHUFFLE DATASET 
# it is arranged by flower types - you want a random sequencing

class_labels = ['Iris-setosa', 'Iris-versicolor', ]
data = data[data['Class'].isin(class_labels)]


X = data[features].values			

le = LabelEncoder()
Y = le.fit_transform(data['Class'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, 
                                                    random_state=3)
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)	

prediction = log_reg_classifier.predict(X_test)
accuracy = np.mean(prediction == Y_test)
print(round(accuracy,2))