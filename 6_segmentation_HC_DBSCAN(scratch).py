import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/ferialnajiantabriz/Desktop/codes/DataMiningPro/heart.csv")

# Check for missing values
print("Checking for missing values:")
print(df.isnull().sum())

# One-Hot Encoding Categorical Variables
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # Avoid dummy variable trap

# Separate features (X), excluding the target variable
X = df.drop(columns=["target"])

# Continuous variables
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Normalize Continuous Variables
from sklearn.preprocessing import StandardScaler

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


# 1. Hierarchical Clustering


def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_distance_matrix(data):
    """Calculate the initial distance matrix for the dataset."""
    n = len(data)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i, j] = euclidean_distance(data[i], data[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


def hierarchical_clustering(data, n_clusters):
    """Perform agglomerative hierarchical clustering using Ward's method."""
    clusters = {i: [i] for i in range(len(data))}
    distance_matrix = calculate_distance_matrix(data)

    while len(clusters) > n_clusters:
        min_distance = float('inf')
        closest_pair = None
        for i in clusters:
            for j in clusters:
                if i != j:
                    cluster_i_points = [data[idx] for idx in clusters[i]]
                    cluster_j_points = [data[idx] for idx in clusters[j]]
                    combined_points = cluster_i_points + cluster_j_points
                    new_centroid = np.mean(combined_points, axis=0)
                    sse = np.sum((combined_points - new_centroid) ** 2)

                    if sse < min_distance:
                        min_distance = sse
                        closest_pair = (i, j)

        i, j = closest_pair

        # Debug: Print the merging clusters
        i, j = closest_pair
        print(f"Merging clusters: {i}, {j}")
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    cluster_labels = np.zeros(len(data), dtype=int)
    for cluster_id, point_indices in clusters.items():
        for idx in point_indices:
            cluster_labels[idx] = cluster_id

    return cluster_labels


n_clusters = 4
cluster_labels_hierarchical = hierarchical_clustering(X_features, n_clusters)
X['Cluster_Hierarchical'] = cluster_labels_hierarchical

print("\nHierarchical Clustering Results:")
print(X['Cluster_Hierarchical'].value_counts())


# 2. DBSCAN

def dbscan(data, eps, min_pts):
    """Perform DBSCAN clustering from scratch."""
    n = len(data)
    cluster_labels = -1 * np.ones(n)  # -1 indicates noise
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_idx, neighbors):
        cluster_labels[point_idx] = cluster_id
        while neighbors:
            neighbor_idx = neighbors.pop()
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)
            if cluster_labels[neighbor_idx] == -1:
                cluster_labels[neighbor_idx] = cluster_id

    for point_idx in range(n):
        if not visited[point_idx]:
            visited[point_idx] = True
            neighbors = region_query(point_idx)
            if len(neighbors) >= min_pts:
                expand_cluster(point_idx, neighbors)
                cluster_id += 1

    return cluster_labels


eps = 0.5
min_pts = 5
cluster_labels_dbscan = dbscan(X_features, eps, min_pts)
X['Cluster_DBSCAN'] = cluster_labels_dbscan

print("\nDBSCAN Clustering Results:")
print(X['Cluster_DBSCAN'].value_counts())


# Save Results

X.to_csv("clustering_results.csv", index=False)
print("\nClustering results saved to 'clustering_results.csv'.")
