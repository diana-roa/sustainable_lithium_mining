import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def k_medoids(X, k, max_iters=100):
    num_samples = X.shape[0]
    
    # Randomly initialize medoids
    medoids = X[np.random.choice(num_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign clusters based on the nearest medoid
        distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_medoids = np.copy(medoids)
        
        # Update medoids
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            
            dist_matrix = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
            total_distances = np.sum(dist_matrix, axis=1)
            medoid_index = np.argmin(total_distances)
            new_medoids[i] = cluster_points[medoid_index]

        # Check for convergence
        if np.all(np.linalg.norm(medoids - new_medoids, axis=1) == 0):
            break

        medoids = new_medoids

    # Assign final clusters
    final_distances = np.linalg.norm(X[:, np.newaxis] - medoids, axis=2)
    final_labels = np.argmin(final_distances, axis=1)

    return medoids, final_labels

# Load the data
df = pd.read_csv(r'C:\Users\b407731\Desktop\Daily_basis_Mwh\DK.csv')

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

# Perform K-Medoids clustering with 3 clusters
k = 3
medoids, cluster_labels = k_medoids(data_clustering_normalized, k)
data_for_clustering['Cluster'] = cluster_labels

# Find the representative year (closest to the cluster centers) for each cluster
representative_years = []
for cluster in range(k):
    cluster_data = data_for_clustering[data_for_clustering['Cluster'] == cluster]
    distances = np.linalg.norm(data_clustering_normalized[cluster_data.index] - medoids[cluster], axis=1)
    medoid_index = np.argmin(distances)
    representative_year = cluster_data.iloc[medoid_index]['Year']
    representative_years.append(representative_year)

# Calculate extreme values (min and max) for average demand
min_demand_before = min(avg_demand_before_norm)
max_demand_before = max(avg_demand_before_norm)
min_demand_after = min(avg_demand_after_norm)
max_demand_after = max(avg_demand_after_norm)

min_year_before = years[avg_demand_before_norm.index(min_demand_before)]
max_year_before = years[avg_demand_before_norm.index(max_demand_before)]
min_year_after = years[avg_demand_after_norm.index(min_demand_after)]
max_year_after = years[avg_demand_after_norm.index(max_demand_after)]

# Print extreme values
print(f"Minimum average demand before normalization: {min_demand_before} in year {min_year_before}")
print(f"Maximum average demand before normalization: {max_demand_before} in year {max_year_before}")
print(f"Minimum average demand after normalization: {min_demand_after} in year {min_year_after}")
print(f"Maximum average demand after normalization: {max_demand_after} in year {max_year_after}")

# Visualize the clustered data and the representative years
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot average demand before normalization on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Demand Before Normalization (MWh)', color=color)
scatter1 = ax1.scatter(years, avg_demand_before_norm, color=color, label='Average Demand Before Normalization')
ax1.tick_params(axis='y', labelcolor=color)

# Highlight the representative years and extreme values
for rep_year in representative_years:
    rep_index = data_for_clustering[data_for_clustering['Year'] == rep_year].index[0]
    ax1.scatter(rep_year, avg_demand_before_norm[rep_index], color='cyan', s=100, edgecolor='k', zorder=5)
ax1.scatter(min_year_before, min_demand_before, color='orange', s=100, edgecolor='k', zorder=5, label='Min Demand Before Norm')
ax1.scatter(max_year_before, max_demand_before, color='green', s=100, edgecolor='k', zorder=5, label='Max Demand Before Norm')

# Create a secondary y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Normalized Average Demand', color=color)
scatter2 = ax2.scatter(years, avg_demand_after_norm, color=color, label='Average Demand After Normalization')
ax2.tick_params(axis='y', labelcolor=color)

# Highlight the representative years and extreme values
for rep_year in representative_years:
    rep_index = data_for_clustering[data_for_clustering['Year'] == rep_year].index[0]
    ax2.scatter(rep_year, avg_demand_after_norm[rep_index], color='magenta', s=100, edgecolor='k', zorder=5)
ax2.scatter(min_year_after, min_demand_after, color='purple', s=100, edgecolor='k', zorder=5, label='Min Demand After Norm')
ax2.scatter(max_year_after, max_demand_after, color='cyan', s=100, edgecolor='k', zorder=5, label='Max Demand After Norm')

# Add trend lines
# Linear regression for average demand before normalization
X = np.array(years).reshape(-1, 1)
y_before = np.array(avg_demand_before_norm)
reg_before = LinearRegression().fit(X, y_before)
trend_line_before = reg_before.predict(X)
ax1.plot(years, trend_line_before, color='darkblue', linestyle='--', linewidth=2, label='Trend Line Before Norm')

# Linear regression for average demand after normalization
y_after = np.array(avg_demand_after_norm)
reg_after = LinearRegression().fit(X, y_after)
trend_line_after = reg_after.predict(X)
ax2.plot(years, trend_line_after, color='darkred', linestyle='--', linewidth=2, label='Trend Line After Norm')

# Add text labels for representative years
for year in representative_years:
    index = years.tolist().index(year)
    ax1.text(year, avg_demand_before_norm[index], f'Rep {year}', fontsize=9, color='cyan', ha='right')
    ax2.text(year, avg_demand_after_norm[index], f'Rep {year}', fontsize=9, color='magenta', ha='right')

# Title and legend
fig.suptitle("Average Demand Before and After Normalization Over Years")
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
fig.tight_layout()

plt.show()
