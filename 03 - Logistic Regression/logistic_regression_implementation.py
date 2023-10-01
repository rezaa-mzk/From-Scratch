# Logistic Regression Implementation From Scratch

## Importing the Libraries

import numpy as np


## Importing the Dataset

from sklearn import datasets
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target


## Splitting the Dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


## Defining the Logistic Regression Class

class CustomLogisticRegression():

    def __init__(self, max_iter = 1000, learning_rate = 0.01):
        
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.theta = None
        self.bias = None
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.theta = np.zeros((n_features, 1))
        self.bias = 0
        
        for i in range(0, self.max_iter):
            weighted_sum = np.dot(self.theta.T, X.T) + self.bias
            h_theta = self._sigmoid(weighted_sum)
            self.gradient_descent(y, h_theta, X, n_samples)
                
    def predict(self, inputs):
        
        return [1 if _ >= 0.5 else 0 for _ in self._sigmoid(np.dot(inputs, self.theta) + self.bias)]
        
    def _sigmoid(self, x):
        
        return 1.0 / (1.0 + np.exp(-x))
    
    def gradient_descent(self, y, h_theta, X, n_samples):
        
        self.theta = self.theta - self.learning_rate * (-1 / n_samples) * np.dot((y - h_theta), X).T
        self.bias = self.bias - self.learning_rate * (-1 / n_samples) * np.sum(y - h_theta)
        


## Fit model and predict test results

clf = CustomLogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


## Check accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

