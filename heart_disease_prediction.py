# -*- coding: utf-8 -*-
"""Heart Disease Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zjRdsG7MZ2DSS3u-nKBrndyC3bpFEQt3

#importing the dependencies
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""#Data collection and Processing"""

#loading the csv data to a pandas dataframe
heart_data = pd.read_csv('heart-disease.csv')

#print first 5 rows of the dataset
heart_data.head()

#@title Default title text
#print last 5 rows of the dataset
heart_data.tail()

#number of rows and columns in the dataset
heart_data.shape

#getting some info about the data
heart_data.info()

#checking for missing values
heart_data.isnull().sum()

#statistical measures about the data
heart_data.describe()

#checking the distribution of target variable
heart_data['target'].value_counts()

"""1--> Defective heart
0--> Healthy heart

#Splitting features and target
"""

X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

print(X)

print(Y)

"""#Splitting  into training data & test data"""

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2, stratify = Y, random_state =2)

print(X.shape,X_train.shape,X_test.shape)

"""#Model training

Logistics Regression
"""

model = LogisticRegression()

#training the LogisticsRegression model with Training data
model.fit(X_train,Y_train)

"""#Model Evaluation

Accuracy score
"""

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ',training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)

print('Accuracy on test data: ',test_data_accuracy)

"""#Building a predictive system"""

input_data = (50,	0,	0,	110,	254,	0,	0,	159,	0,0.0,	2,	0,	2	)

#change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The person does not have a Heart Disease')
else:
  print('The person has Heart Disease')

"""Saving the trained model"""

import pickle

filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))

"""loading the save model"""

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


input_data = (50,	0,	0,	110,	254,	0,	0,	159,	0,0.0,	2,	0,	2	)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
  print('The person does not have a Heart Disease')
else:
  print('The person has Heart Disease')

