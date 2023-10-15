**Simple Linear Regression Implementation From Scratch**

---

### Description

This Python code implements simple linear regression from scratch, a fundamental machine learning algorithm used for predicting a continuous target variable based on a single feature. In this implementation, a custom `CustomSimpleLinearRegression` class is defined, which includes methods for fitting the model, making predictions, and evaluating the performance using the coefficient of determination (R^2 score).

### Files

- **`simple_linear_regression.py`**: Python script containing the implementation of simple linear regression.
- **`simple_linear_regression.ipynb`**: iPython note-book script containing the implementation of simple linear regression.
- **`Salary_Data.csv`**: Dataset containing years of experience and corresponding salaries. This dataset has been obtained from Kaggle to access it you can use this link (https://www.kaggle.com/datasets/vihansp/salary-data).

### Different Parts of Code

1. **Import Libraries**: Importing necessary libraries including NumPy, Pandas, and Matplotlib.

2. **Load Dataset**: Loading the dataset using Pandas. Prepare the feature matrix `X` and the target variable `y` and visualizing its linear nature using matplotlib.pyplot.

3. **Split Data**: Splitting the dataset into training, cross-validation, and test sets using `train_test_split` from scikit-learn.

4. **Simple Linear Regression Class**:
    - Initializing the `CustomSimpleLinearRegression` class with hyperparameters such as maximum iterations (1000) and learning rate (0.01).
    - Fitting the model on the training data.
    - Making predictions on training, cross-validation, and test sets.

5. **Evaluation**: Evaluating the model using the coefficient of determination (R^2 score), a measure of how well the model fits the data.

### Usage

Just clone the code subtitude dataset and run it.

### Author

*Reza Mazaheri Kashani*