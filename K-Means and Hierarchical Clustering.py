
# K-Means and Hierarchical Clustering
# SPPU Machine Learning Practical

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("sales_data_sample.csv", encoding="cp1252")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

# Select important numerical features

features = ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]

X = df[features].copy()

# Handle missing values
X = X.fillna(X.median())

print("\nSelected Features:")
print(X.head())


# --------------------------------------------------
# 3. NORMALIZE DATA
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 4. ELBOW METHOD FOR K-MEANS
# --------------------------------------------------

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)


# Plot Elbow Graph

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()


# --------------------------------------------------
# 5. K-MEANS CLUSTERING
# --------------------------------------------------

# Select K after observing the elbow graph
# Change this value according to your graph

optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Cluster Counts:")
print(df["KMeans_Cluster"].value_counts().sort_index())


# --------------------------------------------------
# 6. HIERARCHICAL CLUSTERING
# --------------------------------------------------

# Create dendrogram using a sample of 100 records

sample_data = X_scaled[:100]

linked = linkage(sample_data, method="ward")

plt.figure(figsize=(12, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()


# Apply Agglomerative Clustering

hc = AgglomerativeClustering(
    n_clusters=optimal_k,
    linkage="ward"
)

df["Hierarchical_Cluster"] = hc.fit_predict(X_scaled)

print("\nHierarchical Cluster Counts:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())


# --------------------------------------------------
# 7. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\nFinal Clustering Results:")

print(df[features + [
    "KMeans_Cluster",
    "Hierarchical_Cluster"
]].head(10))


# --------------------------------------------------
# TIME AND SPACE COMPLEXITY
# --------------------------------------------------

# K-Means Time Complexity: O(I * N * K * F)
# K-Means Space Complexity: O(N * F + K * F)

# Hierarchical Time Complexity: Approximately O(N^2 log N)
# Hierarchical Space Complexity: O(N^2)

# N = Number of data points
# K = Number of clusters
# F = Number of features
# I = Number of K-Means iterations
