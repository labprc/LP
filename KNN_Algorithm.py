
# K-Nearest Neighbors (KNN) Algorithm
# SPPU Machine Learning Practical

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

# --------------------------------------------------
# 1. READ DATASET
# --------------------------------------------------

df = pd.read_csv("diabetes.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

# Outcome is the target column
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# --------------------------------------------------
# 3. NORMALIZE DATA
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# 4. BUILD AND TRAIN KNN MODEL
# --------------------------------------------------

# Initialize KNN with K = 5
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

print("\nKNN Model Training Completed!")


# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate error rate
error_rate = 1 - accuracy

# Calculate precision
precision = precision_score(y_test, y_pred, zero_division=0)

# Calculate recall
recall = recall_score(y_test, y_pred, zero_division=0)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Error Rate:", round(error_rate * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))


# --------------------------------------------------
# 7. CONFUSION MATRIX GRAPH
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()


# --------------------------------------------------
# TIME AND SPACE COMPLEXITY
# --------------------------------------------------

# Training Time Complexity: Approximately O(1)
# Prediction Time Complexity: O(M * N * F)
# M = Number of test samples
# N = Number of training samples
# F = Number of features
#
# Space Complexity: O(N * F)
# N = Number of training samples
# F = Number of features
