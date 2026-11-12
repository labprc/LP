
# Email Spam Classification using KNN and SVM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------------
# 1. Load and Pre-process Dataset
# -----------------------------------------

df = pd.read_csv("emails.csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Remove unnecessary Email No. column
df.drop(columns=["Email No."], inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=["Prediction"])
y = df["Prediction"]

print("\nClass Distribution:")
print(y.value_counts())

print("\nDataset after Pre-processing:")
print("Features:", X.shape)
print("Target:", y.shape)


# -----------------------------------------
# 2. Split Dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# -----------------------------------------
# 3. Feature Scaling
# -----------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------------------
# 4. K-Nearest Neighbors (KNN)
# -----------------------------------------

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("\nKNN Classification Results:")

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)

print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))


# -----------------------------------------
# 5. Support Vector Machine (SVM)
# -----------------------------------------

svm = SVC(kernel="linear")

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("\nSVM Classification Results:")

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))


# -----------------------------------------
# 6. Confusion Matrix
# -----------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.heatmap(
    confusion_matrix(y_test, knn_pred),
    annot=True, fmt="d", cmap="Blues"
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)

sns.heatmap(
    confusion_matrix(y_test, svm_pred),
    annot=True, fmt="d", cmap="Greens"
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# -----------------------------------------
# 7. Compare Both Models
# -----------------------------------------

results = pd.DataFrame({
    "Model": ["KNN", "SVM"],
    "Accuracy": [knn_accuracy, svm_accuracy],
    "Precision": [knn_precision, svm_precision],
    "Recall": [knn_recall, svm_recall],
    "F1 Score": [knn_f1, svm_f1]
})

print("\nModel Performance Comparison:")
print(results)

# Find the better model based on accuracy
if svm_accuracy > knn_accuracy:
    print("\nSVM performs better.")
elif knn_accuracy > svm_accuracy:
    print("\nKNN performs better.")
else:
    print("\nBoth models have equal accuracy.")


# -----------------------------------------
# Time and Space Complexity
# -----------------------------------------

# KNN Training Time: O(1) apart from data storage
# KNN Prediction Time: O(m * n * p)
# SVM Training Time: Depends on solver and data;
# approximately O(n^2 * p) for common kernel SVM solvers
# SVM Prediction Time: Depends on support vectors

# Space Complexity:
# Dataset and scaled data: O(n * p)
# KNN: O(n * p) for storing training data
# SVM: Depends on number of support vectors
# n = records, p = features, m = test records
