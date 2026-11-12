
# Bank Customer Churn Prediction Using ANN
# SPPU Machine Learning Practical

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --------------------------------------------------
# 1. READ DATASET
# --------------------------------------------------

df = pd.read_csv("Churn_Modelling.csv")

print("First Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# Only graphs from reference code
# --------------------------------------------------

# Graph 1: Pair Plot

sns.pairplot(
    df[["CreditScore", "Age", "Balance", "EstimatedSalary", "Exited"]]
    .sample(min(1000, len(df)), random_state=42),
    hue="Exited"
)
plt.show()


# Graph 2: Correlation Heatmap

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()


# Graph 3: Geography vs Churn

sns.countplot(data=df, x="Geography", hue="Exited")
plt.title("Geography vs Churn")
plt.show()


# Graph 4: Gender vs Churn

sns.countplot(data=df, x="Gender", hue="Exited")
plt.title("Gender vs Churn")
plt.show()


# Graph 5: Credit Card vs Churn

sns.countplot(data=df, x="HasCrCard", hue="Exited")
plt.title("Credit Card vs Churn")
plt.show()


# Graph 6: Active Member vs Churn

sns.countplot(data=df, x="IsActiveMember", hue="Exited")
plt.title("Active Member vs Churn")
plt.show()


# Boxplots from reference code

for col in ["Age", "CreditScore", "Balance"]:

    sns.boxplot(data=df, x="Exited", y=col, hue="Gender")

    plt.title(col + " vs Churn and Gender")
    plt.show()


# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

# Remove unnecessary columns

df = df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)

# Input features

X = df.drop("Exited", axis=1)

# Output target

y = df["Exited"]

# Convert categorical columns into numerical values

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)


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

# Fit scaler only on training data

X_train = scaler.fit_transform(X_train)

# Transform testing data

X_test = scaler.transform(X_test)


# --------------------------------------------------
# 4. BUILD AND TRAIN ANN MODEL
# --------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42
)

# Train the model

model.fit(X_train, y_train)

print("\nANN Model Training Completed!")


# --------------------------------------------------
# 5. PREDICTION AND MODEL EVALUATION
# --------------------------------------------------

# Predict test data

y_pred = model.predict(X_test)


# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nANN Model Accuracy:",
      round(accuracy * 100, 2), "%")


# Print confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Classification report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# CONFUSION MATRIX GRAPH
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Churned", "Churned"],
    yticklabels=["Not Churned", "Churned"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# --------------------------------------------------
# TIME AND SPACE COMPLEXITY
# --------------------------------------------------

# Time Complexity:
# Approximately O(E * N * W)
# E = Number of training epochs
# N = Number of training samples
# W = Number of network weights

# Space Complexity:
# Approximately O(N * F + W)
# N = Number of training samples
# F = Number of input features
# W = Number of network weights
