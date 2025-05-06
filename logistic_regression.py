"""EECS545 HW2: Logistic Regression."""

import numpy as np
import math


def hello():
    print('Hello from logistic_regression.py')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def naive_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the datset (X, Y).

    This implementation uses a naive set of nested loops over the data.
    Specifically, we are required to use Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
    """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        grad = np.zeros(d)
        H = np.zeros((d, d))
        for data_x, data_y in zip(X, Y):
            ###################################################################
            pred = sigmoid(np.dot(data_x, w))  # Sigmoid activation
            error = pred - data_y  # Difference between predicted and actual
            
            # Compute gradient
            grad += error * data_x
            
            # Compute Hessian (outer product)
            H += pred * (1 - pred) * np.outer(data_x, data_x)
            ###################################################################
            # raise NotImplementedError("TODO: Add your implementation here.")
            ###################################################################
            #                        END OF YOUR CODE                         #
            ###################################################################
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w


def vectorized_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the dataset (X, Y).

    This implementation will vectorize the implementation in naive_logistic_regression,
    which implements Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
  """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        #######################################################################
        # Compute the predicted probabilities using the sigmoid function
        preds = sigmoid(np.dot(X, w))  # Shape: (N,)
        
        # Compute the gradient (vectorized)
        grad = np.dot(X.T, preds - Y)  # Shape: (d,)
        
        # Compute the Hessian (vectorized)
        diag = preds * (1 - preds)  # Shape: (N,)
        H = np.dot(X.T * diag, X)  # Shape: (d, d)
        #######################################################################
        # grad = None  # hint: grad.shape should be (d, ) at the end of this block
        # H = None  # hint: H.shape should be (d, d) at the end of this block
        # raise NotImplementedError("TODO: Add your implementation here.")
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w


def compute_y_boundary(X_coord: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the matched y coordinate value for the decision boundary from
    the x coordinate and coefficients w.

    Inputs:
      - X_coord: Numpy array of shape (d, ). List of x coordinate values.
      - w: Numpy array of shape (3, ) that stores the coefficients.

    Returns:
      - Y_coord: Numpy array of shape (d, ).
                 List of y coordinate values with respect to the coefficients w.
    """
    Y_coord = None
    ###########################################################################
    w_0, w_1, w_2 = w

    # Compute y values based on the decision boundary equation
    Y_coord = (-w_0 -( w_1 * X_coord)) / w_2
    ###########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return Y_coord
