# Decision Tree Implementation From Scratch

This Python script implements a decision tree from scratch for classification. The decision tree is trained on the Iris dataset, and its performance is evaluated on training, cross-validation, and test sets. The implementation includes the definition of a Node class and a DecisionTree class.

---

## Different Parts of Code

- `Import Libraries`: Importing necessary libraries including NumPy, Collections and Matplotlib.
- `Load Dataset`: Loading the `Iris` dataset using Scikit learns's datasets module.
- `Split Data`: Splitting the dataset into a training set, a cross-validation set, and a test set using `train_test_split` from scikit-learn.
- `Node` class: Represents a node in the decision tree.
- `DecisionTree` class: Implements the decision tree algorithm.
  - `fit(X, y)`: Fits the decision tree to the training data.
  - `predict(X)`: Predicts the labels for the input data.
- `Model instantiation and fitting`: A `DecisionTree` instance is created and fitted to the training data.
- `Predictions`: Making predictions on the training, cross-validation, and test sets.
- `Evaluation`: Evaluating the model's performance using accuracy score and visualize the confusion matrix.

---

## Author

*Reza Mazaheri Kashani*

---