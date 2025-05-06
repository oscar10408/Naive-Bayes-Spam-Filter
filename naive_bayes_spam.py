"""EECS545 HW2: Naive Bayes for Classifying SPAM."""

from typing import Tuple

import numpy as np
import math


def hello():
    print('Hello from naive_bayes_spam.py')


def train_naive_bayes(X: np.ndarray, Y: np.ndarray,
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes probabilities for logit x being each class.

    Inputs:
      - X: Numpy array of shape (num_mails, vocab_size) that represents emails.
        The (i, j)th entry of X represents the number of occurrences of the
        j-th token in the i-th document.
      - Y: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
    Returns: A tuple of
      - mu_spam: Numpy array of shape (vocab_size). mu value for SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). mu value for Non-SPAM mails.
      - phi: the ratio of SPAM mail from the dataset email.
    """
    num_mails, vocab_size = X.shape
    mu_spam = None
    mu_non_spam = None
    phi = 0.0
    ###########################################################################
    phi = np.mean(Y)  # This is the proportion of spam emails

    # Filter X by spam and non-spam emails based on Y
    X_spam = X[Y == 1, :]
    X_non_spam = X[Y == 0, :]

    # Calculate the word counts for spam and non-spam emails
    word_counts_spam = np.sum(X_spam, axis=0)
    word_counts_non_spam = np.sum(X_non_spam, axis=0)

    # Apply Laplace smoothing and calculate the probabilities for each word (mu)
    mu_spam = (word_counts_spam + 1) / (np.sum(word_counts_spam) + vocab_size)
    mu_non_spam = (word_counts_non_spam + 1) / (np.sum(word_counts_non_spam) + vocab_size)
    ###########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return mu_spam, mu_non_spam, phi


def test_naive_bayes(X: np.ndarray,
                     mu_spam: np.ndarray,
                     mu_non_spam: np.ndarray,
                     phi: float,
                     ) -> np.ndarray:
    """Classify whether the emails in the test set is SPAM.

    Inputs:
      - X: Numpy array of shape (num_mails, vocab_size) that represents emails.
        The (i, j)th entry of X represents the number of occurrences of the
        j-th token in the i-th document.
      - mu_spam: Numpy array of shape (vocab_size). mu value for SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). mu value for Non-SPAM mails.
      - phi: the ratio of SPAM mail from the dataset email.
    Returns:
      - pred: Numpy array of shape (num_mails). Mark 1 for the SPAM mails.
    """
    pred = np.zeros(X.shape[0])
    ###########################################################################
    log_phi = np.log(phi)
    log_one_minus_phi = np.log(1 - phi)
    
    log_prob_spam = np.dot(np.log(mu_spam), X.T) + log_phi
    log_prob_non_spam = np.dot(np.log(mu_non_spam), X.T) + log_one_minus_phi
    # Classify the email as spam if log probability of spam is higher
    for i in range(len(pred)):
      if log_prob_spam[i] > log_prob_non_spam[i]:
          pred[i] = 1  # Spam
      else:
          pred[i] = 0  # Non-spam
    ###########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return pred


def evaluate(pred: np.ndarray, Y: np.ndarray) -> float:
    """Compute the accuracy of the predicted output w.r.t the given label.

    Inputs:
      - pred: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
      - Y: Numpy array of shape (num_mails). It includes 0 (non-spam) or 1 (spam).
    Returns:
      - accuracy: accuracy value in the range [0, 1].
    """
    accuracy = np.mean((pred == Y).astype(np.float32))

    return accuracy


def get_indicative_tokens(mu_spam: np.ndarray,
                          mu_non_spam: np.ndarray,
                          top_k: int,
                          ) -> np.ndarray:
    """Filter out the most K indicative vocabs from mu.

    We will check the lob probability of mu's. Your goal is to return `top_k`
    number of vocab indices.

    Inputs:
      - mu_spam: Numpy array of shape (vocab_size). The mu value for
                 SPAM mails.
      - mu_non_spam: Numpy array of shape (vocab_size). The mu value for
                     Non-SPAM mails.
      - top_k: The number of indicative tokens to generate. A positive integer.
    Returns:
      - idx_list: Numpy array of shape (top_k), of type int (or int32).
                  Each index represent the vocab in vocabulary file.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    idx_list = np.zeros(top_k, dtype=np.int32)
    ###################################################################
    # Calculate the log-likelihood ratio for each word
    log_ratio = np.log(mu_spam) - np.log(mu_non_spam)
    idxSort = np.flip(np.argsort(log_ratio))
    # Get the indices of the top_k most indicative tokens based on log_ratio
    idx_list = idxSort[:top_k]
    ###################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return idx_list
