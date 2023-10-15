**Multiple Linear Regression Implementation From Scratch**

---

### Description

This code implements multiple linear regression from scratch in Python. Multiple linear regression is a statistical method to model the relationship between multiple independent variables and a dependent variable by fitting a linear equation to observed data.

### Files

- **`multiple_linear_regression.py`**: Python script containing the implementation of multiple linear regression.
- **`multiple_linear_regression.ipynb`**: iPython note-book script containing the implementation of multiple linear regression.
- **`Student_Performance.csv`**: Dataset containing student performance data. This dataset was obtained from Kaggle. You can find the dataset and its description using this link (https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression).

### Different Parts of Code
This code comprises these parts:

1. **Import Libraries**: Importing Numpy library to handle scientific calculations like matrix multiplication and Pandas to import data from a .csv file into the code.

2. **Import Dataset**: Loading the dataset using Pandas and prepare the feature matrix `X` and the target variable `y`.

3. **Data Preprocessing**:
    - **Categorical Encoding**: Encoding categorical features.
    - **Splitting Data**: Splitting the dataset into training, cross-validation, and test sets.
    - **Feature Scaling**: Standardizing the features to have mean=0 and variance=1.

4. **Multiple Linear Regression Class**:
    - Initializing the class with a learning rate (0.01) and a number of iterations (1000).
    - Fitting the model on the training data.
    - Make predictions on training, cross-validation, and test sets.

5. **Evaluation**: Evaluating the model using R^2 score, a measure of how well the model fits the data.


### Usage

Just clone the code subtitude your own dataset and run the code.

### Author

*Reza Mazaheri Kashani*


---
