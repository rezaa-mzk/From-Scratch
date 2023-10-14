**Softmax Regression Implementation From Scratch**

---

### Description

This repository contains a Python implementation of softmax regression from scratch. Softmax regression is a generalization of logistic regression to multiple classes. It is commonly used for multiclass classification problems. In this implementation, a custom Softmax Regression class is defined and trained on the handwritten digits dataset (MNIST).

### Files

- **`softmax_regression.py`**: Python script containing the implementation of softmax regression.
- **`softmax_regression.ipnyb`**: Python Notebook script containing the implementation of softmax regression.

### Different Parts of Code

This code comprises these parts:

1. **Import Libraries**: Importing the necessary libraries including NumPy and Matplotlib.

2. **Load Dataset**: Importing the dataset using the `datasets` module from scikit-learn. In this example, the Scikit-Learns's MNIST handwritten digits dataset is used.

3. **Data Preprocessing**:
    - **Data Loading**: Loading the dataset and preprocess the features and labels.
    - **Split Data**: Splitting the dataset into training, cross-validation, and test sets.
    - **Label Binarization**: Converting labels into binary vectors (one-hot encoding for target variable).

4. **Custom Softmax Regression Class**:
    - Initializing the class with hyperparameters such as maximum iterations (1000) and learning rate (0.01).
    - Fitting the model on the training data.
    - Making predictions on training, cross-validation, and test sets.

5. **Evaluation**: Evaluating the model using accuracy.

6. **Visualize**: Visualizing the confusion matrix using Matplotlib.

### Usage Example

Just clone the code subtitude dataset and run it.

### Author

*Reza Mazaheri Kashani*
