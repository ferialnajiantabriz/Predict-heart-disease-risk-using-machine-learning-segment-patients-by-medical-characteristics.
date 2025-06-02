import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'path_to_your_data.csv'  # Update with your dataset path
data = pd.read_csv(file_path)

# Extract features (assuming the target column is 'target', adjust if different)
features = data.drop(columns=['target'], errors='ignore')

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Initialize GMM parameters
n_components = 4  # Number of clusters
n_samples, n_features = scaled_features.shape

# Randomly initialize means (mu)
np.random.seed(42)
means = scaled_features[np.random.choice(n_samples, n_components, replace=False)]

# Initialize covariances (Sigma)
covariances = np.array([np.cov(scaled_features.T) for _ in range(n_components)])

# Initialize weights (pi)
weights = np.ones(n_components) / n_components


# Helper functions for Gaussian PDF
def gaussian_pdf(x, mean, cov):
    d = len(mean)
    diff = x - mean
    exp_term = np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff)
    return exp_term / (np.sqrt((2 * np.pi) ** d * np.linalg.det(cov)))


# Expectation-Maximization (EM) loop
tol = 1e-6  # Convergence tolerance
max_iter = 100  # Max iterations
log_likelihood_old = 0

for iteration in range(max_iter):
    # E-step: Calculate responsibilities
    responsibilities = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        total_prob = 0
        for j in range(n_components):
            responsibilities[i, j] = weights[j] * gaussian_pdf(scaled_features[i], means[j], covariances[j])
            total_prob += responsibilities[i, j]
        responsibilities[i, :] /= total_prob  # Normalize to get probabilities

    # M-step: Update the parameters
    # Update weights
    weights = np.mean(responsibilities, axis=0)

    # Update means
    for j in range(n_components):
        means[j] = np.dot(responsibilities[:, j], scaled_features) / np.sum(responsibilities[:, j])

    # Update covariances
    for j in range(n_components):
        diff = scaled_features - means[j]
        covariances[j] = np.dot(responsibilities[:, j] * diff.T, diff) / np.sum(responsibilities[:, j])

    # Compute log likelihood for convergence check
    log_likelihood_new = 0
    for i in range(n_samples):
        total_prob = 0
        for j in range(n_components):
            total_prob += weights[j] * gaussian_pdf(scaled_features[i], means[j], covariances[j])
        log_likelihood_new += np.log(total_prob)

    # Check for convergence
    if np.abs(log_likelihood_new - log_likelihood_old) < tol:
        print(f'Converged after {iteration + 1} iterations')
        break
    log_likelihood_old = log_likelihood_new

# Predict the clusters for each sample
clusters = np.argmax(responsibilities, axis=1)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Save the dataset with clusters
output_path = 'clustered_heart_disease_data_gmm.csv'  # Specify output file name
data.to_csv(output_path, index=False)
print(f"Clustered dataset saved to {output_path}")

# Perform PCA to reduce the feature space to 2D for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = clusters

# Plot PCA components with the cluster labels
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, marker='o', edgecolor='k')
plt.title('PCA of Heart Disease Dataset with 4 GMM Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Count the number of data points in each cluster
cluster_sizes = data['Cluster'].value_counts().sort_index()

# Plot cluster sizes (Bar plot)
plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, color='lightblue')

# Add title and labels
plt.title('Cluster Sizes (GMM Clustering)')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Points')

plt.tight_layout()
plt.show()