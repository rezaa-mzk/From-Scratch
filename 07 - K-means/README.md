# K-means Clustering From Scratch

This repository contains a Python implementation of the K-means clustering algorithm from scratch. K-means is a popular unsupervised machine learning algorithm used for partitioning a dataset into clusters.

---

## Files

- `k_means_clustering.py`: Python script containing the implementation of the K-means clustering algorithm.
- `k_means_clustering.ipynb`: iPython script containing the implementation of the K-means clustering algorithm

---

## Prerequisites

The following code will use the following libraries:

- NumPy
- Matplotlib
- Scikit-learn (only for V Measure Score calculation)

---

## Dataset

The code uses the `make_blobs` function from Scikit-learn to create a synthetic dataset with blobs.

---

## Different parts of code

- Import Libraries: Importing necessary libraries including NumPy, and Matplotlib.
- Making the Dataset: Creating a dataset using the scikit-learn library's make_blob function.
- Defining `CustomKMeans` Class
- Instantiating and fitting the model
- Visualizing Clustering Results
- Calculating V-Measure score: The V-measure is the harmonic mean between homogeneity and completeness:

```math
v = \frac{(1 + beta)* homogeneity * completeness}{(beta * homogeneity + completeness)}
```

Feel free to experiment with different parameters for the `CustomKMeans` class and datasets.

---

## Author

*Reza Mazaheri Kashani*