# Multilayer Perceptron Implementation From Scratch

---

## Description

This Python code implements a Multilayer Perceptron (MLP) from scratch. An MLP is a type of artificial neural network that can learn and make predictions on complex datasets. In this implementation, the MLP is trained on the "Social_Network_Ads.csv" dataset, where the task is to predict whether a user will purchase a product based on age and estimated salary information.

---

## Files

- **`mlp_from_scratch.py`**: Python script containing the implementation of the MLP.
- **`mlp_from_scratch.ipynp`**: iPython note-book script containing the implementation of the MLP.
- **`Social_Network_Ads.csv`**: Dataset containing age, estimated salary, and purchase decision information. This dataset has been obtained from [Kaggle](https://www.kaggle.com). You can download this dataset using the link https://www.kaggle.com/datasets/alirezahasannejad/social-network-ads.

---

## Different Parts of Code

1. **Import Libraries**: Importing necessary libraries including Pandas, NumPy, and Matplotlib.

2. **Load Dataset**: Loading the dataset using Pandas. Prepare the feature matrix `X` and the target variable `y`.

3. **Split Data**: Splitting the dataset into a training set, a cross-validation set, and a test set using `train_test_split` from scikit-learn.

4. **Feature Scaling**: Performing feature scaling to standardize the features.

5. **Instantiate MLP**: Creating an instance of the `MLP` class with the desired architecture (number of inputs, number of nodes in each layer of hidden layer(s), number of outputs).

6. **Train the Model**: Training the MLP using the training set.

7. **Predictions**: Making predictions on the training, cross-validation, and test sets.

8. **Evaluation**: Evaluating the model's performance using accuracy score and visualize the confusion matrix.

---

## Usage Example

```python
# This example has 2 units in input layer 3 units in first hidden layer, 4 units in second hidden layer and 1 unit in output layer.
# Instantiate MLP and train the model
mlp = MLP(num_inputs=2, hidden_layers=[3, 4], num_outputs=1)
mlp.train(X_train, y_train)

```

---

## Author

*Reza Mazaheri Kashani*

---