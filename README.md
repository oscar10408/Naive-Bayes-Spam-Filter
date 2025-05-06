# 🧠 Classic Classification Algorithms from Scratch

This repository contains hands-on implementations of foundational machine learning classification algorithms using **NumPy**. It includes both standalone `.py` modules and demonstration notebooks to visualize and evaluate model behavior.

---

## 📦 Contents

### 1. Logistic Regression (`logistic_regression.py` / `.ipynb`)
- Implements logistic regression via **Newton's Method**.
- Includes both naive (loop-based) and vectorized versions.
- Visualization of decision boundaries.

### 2. Naive Bayes Spam Filter (`naive_bayes_spam.py` / `.ipynb`)
- Classifies emails as spam or non-spam using multinomial Naive Bayes.
- Includes Laplace smoothing and indicative token identification.
- Visual output of classification results and word importance.

**🧮 Training Code with Laplace Smoothing:**

```python
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
```

### 3. Softmax Regression (`softmax_regression.py` / `.ipynb`)
- Multi-class logistic regression using **softmax activation**.
- Trained with gradient ascent on cross-entropy loss.
- Uses one-vs-rest style weight updates and prediction accuracy evaluation.

---

## 🧪 Algorithms Implemented

| Model               | Optimization      | Special Features                          |
|--------------------|-------------------|--------------------------------------------|
| Logistic Regression| Newton's Method   | Decision boundary visualization            |
| Naive Bayes        | Closed-form       | Indicative token extraction for spam data |
| Softmax Regression | Gradient Ascent   | Multi-class classification (e.g. digits)  |

---

## 🧰 Requirements

- Python 3.7+
- NumPy
- Jupyter Notebook (for `.ipynb` usage)

Install dependencies:
```bash
pip install numpy notebook
