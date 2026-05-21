# Logistic Regression From Scratch

A complete implementation of Logistic Regression from scratch using NumPy for binary classification problems.  
This project demonstrates how Logistic Regression works internally using Gradient Descent and the Sigmoid Function without relying on Scikit-learn for model training.

---

# Features

- Logistic Regression implemented from scratch
- Gradient Descent optimization
- Sigmoid activation function
- Binary Classification
- Data Visualization using Matplotlib & Seaborn
- Evaluation Metrics
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Comparison with Scikit-learn Logistic Regression
- Beginner-friendly code structure

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# Project Structure

```bash
Logistic-Regression-From-Scratch/
│
├── Logistic_Regression.ipynb
├── README.md
└── dataset.csv
```

---

# Logistic Regression Formula

Sigmoid Function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Prediction Formula:

\[
z = wx + b
\]

Gradient Descent Update:

\[
w = w - \alpha \cdot dw
\]

\[
b = b - \alpha \cdot db
\]

---

# Dataset

This project uses a binary classification dataset from Kaggle.

Example datasets:
- Student Pass/Fail Prediction
- Exam Performance Dataset
- Breast Cancer Dataset

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Logistic-Regression-From-Scratch.git
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

# Usage

Run the notebook:

```bash
jupyter notebook
```

Open:

```bash
Logistic_Regression.ipynb
```

---

# Example Code

```python
model = LogisticRegression(lr=0.01, no_of_it=1000)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
```

---

# Evaluation Metrics

```python
print("Accuracy :", accuracy_score(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred))
print("Recall   :", recall_score(Y_test, y_pred))
print("F1 Score :", f1_score(Y_test, y_pred))
```

---

# Data Visualization

Scatter Plot:

```python
sns.scatterplot(
    data=data,
    x='Study Hours',
    y='Previous Scores',
    hue='Pass/Fail'
)
```

Distribution Plot:

```python
sns.histplot(
    data=data,
    x='Study Hours',
    hue='Pass/Fail',
    kde=True
)
```

---

# Comparison with Scikit-learn

This project also compares:
- Custom Logistic Regression
- Scikit-learn Logistic Regression

to understand:
- Optimization
- Gradient Descent
- Numerical Stability
- Regularization

---

# Learning Outcomes

By completing this project, you will understand:
- How Logistic Regression works internally
- Gradient Descent optimization
- Binary classification
- Sigmoid activation
- Evaluation metrics
- Machine Learning workflow

---

# Future Improvements

- Add Regularization (L1/L2)
- Add Multi-class Classification
- Improve Optimization
- Add Decision Boundary Visualization
- Deploy using Flask/Streamlit

---


# License

This project is open-source and available under the MIT License.
