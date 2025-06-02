import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("/Users/ferialnajiantabriz/Desktop/codes/DataMiningPro/heart.csv")

# Check for missing values
print("Checking for missing values:")
print(df.isnull().sum())

# One-Hot Encoding Categorical Variables
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X), excluding the target variable
X = df.drop(columns=["target"])

# Continuous variables
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Normalize Continuous Variables
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# Remove outliers from continuous variables simultaneously
def remove_outliers(data, columns):

    mask = pd.Series(True, index=data.index)
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask &= (data[column] >= lower_bound) & (data[column] <= upper_bound)
    return data[mask]

X = remove_outliers(X, continuous_cols)

# Convert the preprocessed data to a NumPy array for clustering
X_features = X.to_numpy()

print("Preprocessing complete. Dataset ready for segmentation.")

# Custom K-means implementation without using np.sqrt
def kmeans_from_scratch(data, n_clusters, max_iters=100, tol=1e-4):

    np.random.seed(123)
    random_indices = np.random.choice(data.shape[0], size=n_clusters, replace=False)
    centroids = data[random_indices]

    for iteration in range(max_iters):
        # Compute squared Euclidean distances manually without np.sqrt
        diff = data[:, np.newaxis, :] - centroids  # Shape: (n_samples, n_clusters, n_features)
        squared_diff = diff ** 2  # Element-wise square
        sum_squared_diff = squared_diff.sum(axis=2)  # Sum over features -> Shape: (n_samples, n_clusters)

        # Assign clusters based on closest centroid
        cluster_labels = np.argmin(sum_squared_diff, axis=1)

        # Compute new centroids
        new_centroids = np.array([
            data[cluster_labels == i].mean(axis=0) if np.any(cluster_labels == i) else centroids[i]
            for i in range(n_clusters)
        ])

        # Compute SSE
        sse = np.sum([
            np.sum((data[cluster_labels == i] - centroids[i]) ** 2)
            for i in range(n_clusters)
        ])

        # Compute centroid shift
        centroid_shift = np.sum((new_centroids - centroids) ** 2)

        # Check for convergence
        if centroid_shift < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids

    return cluster_labels, centroids, sse

# Determine the optimal number of clusters using the Elbow Method
k_values = range(2, 11)
total_sse = []

print("\nCalculating SSE for different values of K:")
for k in k_values:
    cluster_labels, centroids, sse = kmeans_from_scratch(X_features, n_clusters=k)
    total_sse.append(sse)
    print(f"K = {k}, SSE = {sse}")

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(list(k_values), total_sse, marker="o", linestyle="--")
plt.axvline(x=4, color="red", linestyle="--", label="Optimal K=4")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Total Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal K")
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

print("\nBased on the Elbow Method, the optimal K can be identified visually where the reduction in SSE slows down.")

# Visualize the clusters for the chosen optimal K
optimal_k = 4
cluster_labels, centroids, sse = kmeans_from_scratch(X_features, n_clusters=optimal_k)

# Add cluster labels to the DataFrame for visualization
X_visual = X.copy()
X_visual['Cluster'] = cluster_labels

# Select a subset of features for pairplot visualization to avoid high dimensionality
selected_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sns.pairplot(X_visual, hue='Cluster', vars=selected_vars, palette='Set2', plot_kws={'alpha': 0.7})
plt.suptitle(f'Pairwise Relationships for Clusters (K={optimal_k})', y=1.02)
plt.tight_layout()
plt.show()

# Add Silhouette Score and Silhouette Plot

from sklearn.metrics import silhouette_samples, silhouette_score

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X_features, cluster_labels)
silhouette_vals = silhouette_samples(X_features, cluster_labels)

print(f"\nAverage Silhouette Score for K={optimal_k}: {silhouette_avg}")

# Silhouette Plot
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(optimal_k):
    ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    ith_cluster_silhouette_vals.sort()
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.title("Silhouette Plot for K-Means Clustering")
plt.show()

# Add Pairwise Scatterplots for Clusters
sns.pairplot(X_visual, hue='Cluster', vars=selected_vars, palette='Set2', plot_kws={'alpha': 0.7})
plt.suptitle(f'Pairwise Scatterplots for Clusters (K={optimal_k})', y=1.02)
plt.tight_layout()
plt.show()


# Dynamically update categorical_cols after one-hot encoding
categorical_cols = [col for col in X_visual.columns if col.startswith(('sex_', 'cp_', 'fbs_', 'restecg_', 'exang_', 'slope_', 'ca_', 'thal_'))]

# Bar plots for categorical features
one_hot_categorical_cols = [col for col in X_visual.columns if col.startswith(('sex_', 'cp_', 'fbs_', 'restecg_', 'exang_', 'slope_', 'ca_', 'thal_'))]

for col in one_hot_categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=X_visual, x='Cluster', y=col, errorbar=None, hue='Cluster', dodge=False)
    plt.title(f'Cluster Distribution for {col}')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()



# Analyze continuous variables (mean, standard deviation) for each cluster
cluster_analysis = X_visual.groupby('Cluster').agg({
    'age': ['mean', 'std'],
    'trestbps': ['mean', 'std'],
    'chol': ['mean', 'std'],
    'thalach': ['mean', 'std'],
    'oldpeak': ['mean', 'std']
})

print("\nCluster Analysis (Continuous Variables):")
print(cluster_analysis)

# Analyze categorical features (proportions within each cluster)
categorical_analysis = X_visual.groupby('Cluster')[categorical_cols].mean()
print("\nCluster Analysis (Categorical Variables):")
print(categorical_analysis)

# Save analysis to CSV for documentation
cluster_analysis.to_csv("cluster_continuous_analysis.csv")
categorical_analysis.to_csv("cluster_categorical_analysis.csv")

# Define targeted advice for each cluster based on analysis
cluster_advice = {
    0: "Cluster 0: Older patients with elevated cholesterol levels and higher resting blood pressure. Advice: Focus on low-cholesterol diets, blood pressure management, and regular checkups.",
    1: "Cluster 1: Younger patients with normal cholesterol but moderate exercise capacity. Advice: Encourage physical activity and healthy eating habits.",
    2: "Cluster 2: Patients with lower heart rates and normal resting blood pressure. Advice: General lifestyle recommendations with routine monitoring.",
    3: "Cluster 3: Mixed group with higher oldpeak values and diverse risk factors. Advice: Tailored interventions based on individual needs."
}

# Map advice to each cluster
X_visual['Advice'] = X_visual['Cluster'].map(cluster_advice)

# Save segmented dataset with advice to a CSV file
X_visual.to_csv("segmented_patients_with_advice.csv", index=False)

# Print cluster-wise advice summary
print("\nCluster-Specific Advice:")
for cluster_id, advice in cluster_advice.items():
    print(f"Cluster {cluster_id}: {advice}")
# Bar plot for the number of data points in each cluster
cluster_sizes = X_visual['Cluster'].value_counts()

plt.figure(figsize=(8, 6))
cluster_sizes.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Number of Data Points in Each Cluster (K-Means Clustering)", fontsize=14)
plt.xlabel("Cluster Label", fontsize=12)
plt.ylabel("Number of Data Points", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Perform PCA to reduce the dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_features)

# Add the PCA components to the DataFrame for visualization
X_visual['PCA1'] = pca_result[:, 0]
X_visual['PCA2'] = pca_result[:, 1]

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', data=X_visual, palette='Set2', s=100, alpha=0.7, edgecolor="k"
)
plt.title("Scatter Plot of Clusters (K-Means Clustering)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Cluster", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()