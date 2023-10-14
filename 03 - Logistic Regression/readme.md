**Logistic Regression Implementation From Scratch**

---

### Description

This code implements logistic regression from scratch in Python. Logistic regression is a binary classification algorithm that models the probability of a binary outcome. In this implementation, a custom Logistic Regression class is defined and used to fit and predict on Scikit Learn's breast cancer dataset.

### Files

- **`logistic_regression.py`**: Python script containing the implementation of logistic regression.
- **`logistic_regression.ipnyb`**: Python note book script containing the implementation of logistic regression.

### Different Parts of Code

This code comprises these parts:

1. **Import Libraries**: Importing the necessary libraries including NumPy and Matplotlib.

2. **Load Dataset**: Importing the dataset using the `datasets` module from scikit-learn. In this example, the breast cancer dataset is used.

3. **Data Preprocessing**:
    - **Split Data**: Split the dataset into training, cross-validation, and test sets.
    - **Feature Scaling**: Standardize the features using the StandardScaler.

4. **Custom Logistic Regression Class**:
    - Initializing the class with hyperparameters such as maximum iterations (1000) and learning rate (0.01).
    - Fitting the model on the training data.
    - Making predictions on training, cross-validation, and test sets.

5. **Evaluation**: Evaluating the model using accuracy and confusion matrix.

6. **Visualize**: Visualize the confusion matrix using Matplotlib.

### Usage

Just clone the code subtitude your own dataset and run the code.

### Author

*Reza Mazaheri Kashani*
