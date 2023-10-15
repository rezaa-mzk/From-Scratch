#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression  Implementation From Scratch

# ## Importing Necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing the Dataset

# In[2]:


data = pd.read_csv('Student_Performance.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# ## Encoding the Categorical Feature "Extracurricular Activities"

# In[3]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('ohe', OneHotEncoder(), [2])], remainder='passthrough')
X = ct.fit_transform(X)[:, 1:]      # Deleting one column to avoid dummy trap


# ## Splitting the Dataset into Training, Cross Validation and Test sets

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_, y_train, y_ = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_CV, X_test, y_CV, y_test = train_test_split(X_, y_, test_size = 0.5, random_state = 42)


# ## Feature Scaling

# In[6]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_CV = sc_X.transform(X_CV)
X_test = sc_X.transform(X_test)


# ## Defining Multiple Linear Regression Class

# In[7]:


class MultipleLinearRegression():
    
    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        self.lr =  learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(0, self.n_iter):
            
            y_pred = np.dot(self.weights, X.T) + self.bias
            
            dw = (1/n_samples) * np.dot((y_pred - y), X)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self, X):
        
        return np.dot(self.weights, X.T) + self.bias


# ## Creating Instance and Fitting the Model

# In[8]:


mlr = MultipleLinearRegression()
mlr.fit(X_train, y_train)


# ## Making Predictions

# In[9]:


y_train_pred = mlr.predict(X_train)
y_CV_pred = mlr.predict(X_CV)
y_test_pred = mlr.predict(X_test)


# ## Evaluate the model and displaying the confusion matrix

# In[10]:


from sklearn.metrics import r2_score
print("Training set r^2 score:            ", r2_score(y_train, y_train_pred))
print("Cross validation set r^2 score:    ", r2_score(y_CV, y_CV_pred))
print("Test set r^2 score:                ", r2_score(y_test, y_test_pred))

