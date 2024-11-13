import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Load the data
df = pd.read_csv(r'C:\Git_Trainee_Program\Climate_Year_Historical_analysis\Inputs\Daily_basis_Mwh\SPV.csv')
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract year and month from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Drop the years 1979 and 2024
df = df[~df['Year'].isin([1979, 2024])]

# List of countries (assuming columns are named after countries)
countries = [col for col in df.columns if col not in ['Date', 'Year', 'Month']]

# Calculate the mean, min, and max across all countries for each year
yearly_means = df.groupby('Year')[countries].mean().mean(axis=1)
yearly_min = df.groupby('Year')[countries].mean().min(axis=1)
yearly_max = df.groupby('Year')[countries].mean().max(axis=1)

# Identify the representative year as the year closest to the overall mean
overall_mean = yearly_means.mean()
representative_year = yearly_means.sub(overall_mean).abs().idxmin()

# Identify extreme years based on the minimum and maximum yearly averages
extreme_min = [yearly_means.idxmin()]
extreme_max = [yearly_means.idxmax()]

for country in countries:
    # Check if the entire column is zero
    if df[country].sum() == 0:
        continue  # Skip this country if all values are zero

    # Group by Year and Month, then calculate the average power for each month
    monthly_avg = df.groupby(['Year', 'Month'])[country].mean().unstack(0)

    # Smoothing the data using a rolling mean with a window of 2 months
    monthly_avg_smooth = monthly_avg.rolling(window=2, min_periods=1, center=True).mean()

    # Plotting
    plt.figure(figsize=(12, 8))

    # Define specific colors for the years 1995, 2008, and 2009
    highlighted_years = {1995: 'blue', 2008: 'violet', 2009: 'orange'}

    # Iterate over each year in the dataset
    for year in monthly_avg_smooth.columns:
        if year == representative_year:
            plt.plot(monthly_avg_smooth.index, monthly_avg_smooth[year], label=f"{year} (Representative)",
                     linewidth=2.5, color='black', marker='o')
        elif year in extreme_min:
            plt.plot(monthly_avg_smooth.index, monthly_avg_smooth[year], label=f"{year} (Extreme Min)", linewidth=2.5,
                     color='red', marker='x')
        elif year in extreme_max:
            plt.plot(monthly_avg_smooth.index, monthly_avg_smooth[year], label=f"{year} (Extreme Max)", linewidth=2.5,
                     color='green', marker='h')
        elif year in highlighted_years:
            plt.plot(monthly_avg_smooth.index, monthly_avg_smooth[year], label=f"{year}", linewidth=1,
                     color=highlighted_years[year], marker='D', alpha=0.4)  # Different colors for 1995, 2008, 2009
        else:
            plt.plot(monthly_avg_smooth.index, monthly_avg_smooth[year], color='gray', alpha=0.1)

    # Add labels, legend, and grid
    plt.xlabel('Month')
    plt.ylabel(f'Average Power (MW) Generation in {country}')
    plt.title(f'Monthly SPV in {country} (1980-2023)')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)  # Adding grid for better readability
    plt.tight_layout()  # Adjust layout for better display
    plt.show()

#######################




# Initialize lists to store raw and normalized average Residual Load
avg_RL_before_norm = []
avg_RL_after_norm = []

# Set 'Year' as the first column
df = df[['Year'] + [col for col in df.columns if col != 'Year']]

years = df['Year'].unique()

# Iterate over each year
for year in years:
    df_year = df[df['Year'] == year].copy()

    # Drop the 'Date' and 'Year' and 'Month' columns for normalization
    data = df_year.drop(columns=['Date'])
    data = df_year.drop(columns=['Year'])
    data = df_year.drop(columns=['Month'])

    # Separate the datetime columns from the numerical columns
    numerical_data = data.select_dtypes(include=['number'])

    # Handle missing values by filling them with zeros
    numerical_data.fillna(0, inplace=True)

    # Calculate the mean of the numerical columns
    avg_RL_before_norm.append(numerical_data.mean().mean())

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(numerical_data)

    # Store the average RL for plotting
    avg_RL_after_norm.append(data_normalized.mean().mean())

# Combine years with their corresponding average RL into a DataFrame
data_for_clustering = pd.DataFrame({
    'Year': years,
    'Avg_RL_Before_Norm': avg_RL_before_norm,
    'Avg_RL_After_Norm': avg_RL_after_norm
})

# No need to normalize 'data_for_clustering' again
data_clustering_normalized = data_for_clustering[['Avg_RL_After_Norm']].values


def k_medoids(X, k, max_iters=100, n_init=10):
    best_medoids = None
    best_labels = None
    best_score = -1

    for _ in range(n_init):
        num_samples = X.shape[0]

        # Randomly initialize medoids
        np.random.seed(100)  # This ensures that the same random numbers are generated each time the code runs, leading to consistent results.
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

        # Compute silhouette score
        score = silhouette_score(X, final_labels)
        if score > best_score:
            best_score = score
            best_medoids = medoids
            best_labels = final_labels

    return best_medoids, best_labels


# Define the range of possible cluster numbers
cluster_range = range(3, 6)  # Adjust this range as needed

# Initialize variables to store the best results
best_score = -1
best_k = 0
best_medoids = None
best_cluster_labels = None

# Loop over the range of cluster numbers to find the optimal number of clusters
for k in cluster_range:
    # Perform K-Medoids clustering with k clusters
    medoids, cluster_labels = k_medoids(data_clustering_normalized, k)

    # Compute the silhouette score
    silhouette_avg = silhouette_score(data_clustering_normalized, cluster_labels)
    print(f"Number of clusters: {k}, Average Silhouette Score: {silhouette_avg:.2f}")

    # Update the best score and number of clusters if needed
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k
        best_medoids = medoids
        best_cluster_labels = cluster_labels

# Print the optimal number of clusters
print(f"Optimal number of clusters based on silhouette score: {best_k}")

# Perform K-Medoids clustering with the optimal number of clusters
k = best_k
data_for_clustering['Cluster'] = best_cluster_labels

# Find the representative year (closest to the cluster centers) for each cluster
representative_years = []
for cluster in range(k):
    cluster_data = data_for_clustering[data_for_clustering['Cluster'] == cluster]
    distances = np.linalg.norm(data_clustering_normalized[cluster_data.index] - best_medoids[cluster], axis=1)
    medoid_index = np.argmin(distances)
    representative_year = cluster_data.iloc[medoid_index]['Year']
    representative_years.append(representative_year)

# Print the representative years
print(f"Representative Years: {representative_years}")


# Visualize the clustered data and the representative years
fig, ax = plt.subplots(figsize=(14, 6))

# Use a colormap with 'k' colors for clustering visualization
colors = plt.get_cmap('tab10', k)
for cluster in range(k):
    cluster_data = data_for_clustering[data_for_clustering['Cluster'] == cluster]
    ax.scatter(cluster_data['Year'], cluster_data['Avg_RL_After_Norm'], color=colors(cluster),
               label=f'Cluster {cluster + 1}')

# Highlight the representative years
for rep_year in representative_years:
    rep_index = data_for_clustering[data_for_clustering['Year'] == rep_year].index[0]
    ax.scatter(rep_year, avg_RL_after_norm[rep_index], color='magenta', s=100, edgecolor='k', zorder=5)

# Label all years with smaller font for non-representative years
for i, year in enumerate(years):
    if year in representative_years:
        ax.text(year, avg_RL_after_norm[i], f'Rep {year}', fontsize=9, color='magenta', ha='right')
    else:
        ax.text(year, avg_RL_after_norm[i], str(year), fontsize=7, color='black', ha='right')

# Add trendline for the normalized average RE generation
X = np.array(years).reshape(-1, 1)
y = np.array(avg_RL_after_norm)
reg = LinearRegression().fit(X, y)
trend_line = reg.predict(X)
ax.plot(years, trend_line, color='darkred', linestyle='--', linewidth=2, label='Trend Line')

# Title and legend
fig.suptitle("Normalized Average Hydro Run of River Over Years with K-Medoids Clustering")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()

plt.show()

