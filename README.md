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

![Naive Bayes Spam Classification](./naive_bayes_spam.png)

> 📌 The figure above shows predicted classifications and indicative tokens ranked by log-likelihood ratios.

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
