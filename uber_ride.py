
# Uber Fare Prediction using Machine Learning
# Linear Regression and Random Forest Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------------------
# 1. Load and Pre-process Dataset
# -----------------------------------------

df = pd.read_csv("uber.csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

# Remove unnecessary columns
df.drop(columns=["Unnamed: 0", "key"],
        errors="ignore", inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Convert datetime
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)

df.dropna(subset=["pickup_datetime"], inplace=True)

# Extract date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["weekday"] = df["pickup_datetime"].dt.weekday

df.drop(columns=["pickup_datetime"], inplace=True)

# Remove invalid values
df = df[
    (df["fare_amount"] > 0) &
    (df["passenger_count"] >= 1) &
    (df["passenger_count"] <= 6) &
    (df["pickup_longitude"].between(-75, -72)) &
    (df["dropoff_longitude"].between(-75, -72)) &
    (df["pickup_latitude"].between(40, 42)) &
    (df["dropoff_latitude"].between(40, 42))
].copy()


# -----------------------------------------
# Calculate Distance using Haversine
# -----------------------------------------

lat1 = np.radians(df["pickup_latitude"])
lon1 = np.radians(df["pickup_longitude"])

lat2 = np.radians(df["dropoff_latitude"])
lon2 = np.radians(df["dropoff_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (np.sin(dlat / 2) ** 2 +
     np.cos(lat1) * np.cos(lat2) *
     np.sin(dlon / 2) ** 2)

c = 2 * np.arcsin(np.sqrt(a))

df["distance"] = 6371 * c

print("\nDataset after Pre-processing:")
print(df.head())
print("Total Records:", len(df))


# -----------------------------------------
# 2. Identify and Remove Outliers
# -----------------------------------------

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["fare_amount"] < lower) |
    (df["fare_amount"] > upper)
]

print("\nNumber of Outliers:", len(outliers))

# Display boxplot
sns.boxplot(x=df["fare_amount"])
plt.title("Fare Amount Before Removing Outliers")
plt.show()

# Remove outliers
df = df[
    (df["fare_amount"] >= lower) &
    (df["fare_amount"] <= upper)
].copy()

print("Records after Removing Outliers:", len(df))


# -----------------------------------------
# 3. Check Correlation
# -----------------------------------------

correlation = df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(correlation, annot=True,
            cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap")
plt.show()


# -----------------------------------------
# 4. Prepare Features and Target
# -----------------------------------------

X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# -----------------------------------------
# 5. Linear Regression
# -----------------------------------------

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)

print("\nLinear Regression Results:")
print("R2 Score:", lr_r2)
print("RMSE:", lr_rmse)
print("MAE:", lr_mae)


# -----------------------------------------
# 6. Random Forest Regression
# -----------------------------------------

rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)

print("\nRandom Forest Regression Results:")
print("R2 Score:", rf_r2)
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)


# -----------------------------------------
# 7. Compare Both Models
# -----------------------------------------

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [lr_r2, rf_r2],
    "RMSE": [lr_rmse, rf_rmse],
    "MAE": [lr_mae, rf_mae]
})

print("\nModel Comparison:")
print(results)

if rf_r2 > lr_r2:
    print("\nRandom Forest performs better.")
else:
    print("\nLinear Regression performs better.")


# -----------------------------------------
# Time and Space Complexity
# -----------------------------------------

# Preprocessing: O(n)
# Distance calculation: O(n)
# Correlation: O(n * p^2)
# Linear Regression: approximately O(n*p^2 + p^3)
# Random Forest: approximately O(T*n*log(n)*p)
# Random Forest worst-case depends on tree depth.

# Space Complexity:
# Dataset and preprocessing: O(n*p)
# Linear Regression: O(p^2) model storage
# Random Forest: O(T * total tree nodes)
# Here n = records, p = features, T = number of trees.
