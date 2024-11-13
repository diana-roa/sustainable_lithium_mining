import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Load the data
df = pd.read_csv(r'C:\Users\b407731\Desktop\Daily_basis_Mwh\Demand.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Calculate the differences between consecutive days for each feature
diffs = df.set_index('Date').diff()

# Find the day with the highest negative change for each feature
max_negative_changes = diffs.idxmin()

# Find the feature with the overall highest negative change
overall_max_negative_change = diffs.min().min()
overall_max_negative_change_day = diffs.stack().idxmin()

# Print the results
print("Day with the highest negative change for each feature:")
print(max_negative_changes)
print("\nOverall highest negative change:")
print(overall_max_negative_change)
print("Day with the overall highest negative change:")
print(overall_max_negative_change_day)

# Preprocess the data for clustering
# Drop the 'Date' column for clustering
data = df.drop(columns=['Date'])

# Handle missing values by filling them with the mean of each column
data.fillna(data.mean(), inplace=True)

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Create a DataFrame from the normalized data
df_normalized = pd.DataFrame(data_normalized, columns=data.columns, index=df['Date'])

# Calculate the differences between consecutive days for each feature in the normalized data
diffs_norm = df_normalized.diff()

# Find the day with the highest negative change for each feature
max_negative_changes_norm = diffs_norm.idxmin()

# Find the feature with the overall highest negative change
overall_max_negative_change_norm = diffs_norm.min().min()
overall_max_negative_change_day_norm = diffs_norm.stack().idxmin()

# Print the results for normalized data
print("Day with the highest negative change for each feature AFTER NORM:")
print(max_negative_changes_norm)
print("\nOverall highest negative change AFTER NORM:")
print(overall_max_negative_change_norm)
print("Day with the overall highest negative change AFTER NORM:")
print(overall_max_negative_change_day_norm)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_normalized)

# Add the cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Find the representative day (medoid) for each cluster
representative_days = []
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    centroid = kmeans.cluster_centers_[cluster]
    distances = np.linalg.norm(scaler.transform(cluster_data.drop(columns=['Date', 'Cluster'])) - centroid, axis=1)
    representative_day = cluster_data.iloc[distances.argmin()]['Date']
    representative_days.append(representative_day)

# Print the representative days for each cluster
print("Representative days for each cluster:")
print(representative_days)

# Calculate the silhouette score to determine the optimal number of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_normalized)
    silhouette_avg = silhouette_score(data_normalized, cluster_labels)
    silhouette_scores.append((n_clusters, silhouette_avg))

# Print silhouette scores for different numbers of clusters
print("Silhouette scores for different numbers of clusters:")
for n_clusters, score in silhouette_scores:
    print(f"Number of clusters: {n_clusters}, Silhouette score: {score}")

# Determine the optimal number of clusters based on the highest silhouette score
optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
print(f"Optimal number of clusters: {optimal_clusters}")

# Refit K-means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(data_normalized)
df['OptimalCluster'] = kmeans.labels_

# Find the representative days (medoids) for each optimal cluster
optimal_representative_days = []
for cluster in range(optimal_clusters):
    cluster_data = df[df['OptimalCluster'] == cluster]
    centroid = kmeans.cluster_centers_[cluster]
    distances = np.linalg.norm(scaler.transform(cluster_data.drop(columns=['Date', 'Cluster', 'OptimalCluster'])) - centroid, axis=1)
    representative_day = cluster_data.iloc[distances.argmin()]['Date']
    optimal_representative_days.append(representative_day)

# Print the representative days for each optimal cluster
print("Representative days for each optimal cluster:")
print(optimal_representative_days)

# Print the size of each optimal cluster
print("Size of each optimal cluster:")
print(df['OptimalCluster'].value_counts())
