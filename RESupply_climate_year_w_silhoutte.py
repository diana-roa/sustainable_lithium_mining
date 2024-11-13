# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:05:53 2024

@author: B407731
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the data
df = pd.read_csv(r'C:\Users\b407731\Desktop\Daily_basis_Mwh\Demand.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract year from the 'Date' column
df['Year'] = df['Date'].dt.year

# Set 'Year' as the first column
df = df[['Year'] + [col for col in df.columns if col != 'Year']]

years = df['Year'].unique()

# Initialize lists to store data for plotting
avg_demand_before_norm = []
avg_demand_after_norm = []

for year in years:
    df_year = df[df['Year'] == year].copy()

    # Drop the 'Date' and 'Year' columns for normalization
    data = df_year.drop(columns=['Date', 'Year'])

    # Handle missing values by filling them with the mean of each column
    data.fillna(data.mean(), inplace=True)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Store the average demand for plotting
    avg_demand_before_norm.append(data.mean().mean())
    avg_demand_after_norm.append(data_normalized.mean().mean())

# Combine years with their corresponding average demands into a DataFrame
data_for_clustering = pd.DataFrame({
    'Year': years,
    'Avg_Demand_Before_Norm': avg_demand_before_norm,
    'Avg_Demand_After_Norm': avg_demand_after_norm
})

# Normalize the data_for_clustering
scaler = StandardScaler()
data_clustering_normalized = scaler.fit_transform(data_for_clustering[['Avg_Demand_Before_Norm', 'Avg_Demand_After_Norm']])

# Perform silhouette analysis for a range of cluster numbers
range_n_clusters = list(range(2, 11))
silhouette_avg_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_clustering_normalized)
    silhouette_avg = silhouette_score(data_clustering_normalized, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is {silhouette_avg}")

# Plot the silhouette scores for different numbers of clusters
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis for KMeans Clustering')
plt.grid(True)
plt.show()

# Determine the optimal number of clusters based on the highest silhouette score
optimal_clusters = range_n_clusters[silhouette_avg_scores.index(max(silhouette_avg_scores))]
print(f"Optimal number of clusters based on silhouette analysis: {optimal_clusters}")

# Apply k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(data_clustering_normalized)
data_for_clustering['Cluster'] = cluster_labels

# Find the representative year (closest to the cluster centers) for each cluster
representative_years = []
for cluster in range(optimal_clusters):
    cluster_data = data_for_clustering[data_for_clustering['Cluster'] == cluster]
    distances = np.linalg.norm(scaler.transform(cluster_data[['Avg_Demand_Before_Norm', 'Avg_Demand_After_Norm']]) - kmeans.cluster_centers_[cluster], axis=1)
    medoid_index = np.argmin(distances)
    representative_year = cluster_data.iloc[medoid_index]['Year']
    representative_years.append(representative_year)

# Print the representative years
print(f"Representative Years: {representative_years}")

# Visualize the clustered data and the representative years
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot average demand before normalization on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Demand Before Normalization (MWh)', color=color)
scatter1 = ax1.scatter(years, avg_demand_before_norm, color=color, label='Average Demand Before Normalization')
ax1.tick_params(axis='y', labelcolor=color)

# Highlight the representative years
for rep_year in representative_years:
    rep_index = data_for_clustering[data_for_clustering['Year'] == rep_year].index[0]
    ax1.scatter(rep_year, avg_demand_before_norm[rep_index], color='cyan', s=100, edgecolor='k', zorder=5)

# Create a secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Normalized Average Demand', color=color)
scatter2 = ax2.scatter(years, avg_demand_after_norm, color=color, label='Average Demand After Normalization')
ax2.tick_params(axis='y', labelcolor=color)

# Highlight the representative years
for rep_year in representative_years:
    rep_index = data_for_clustering[data_for_clustering['Year'] == rep_year].index[0]
    ax2.scatter(rep_year, avg_demand_after_norm[rep_index], color='magenta', s=100, edgecolor='k', zorder=5)

# Title and legend
fig.suptitle("Average Demand Before and After Normalization Over Years")
fig.legend(handles=[scatter1, scatter2], loc="upper left", bbox_to_anchor=(0.1,0.9))
fig.tight_layout()

plt.show()
