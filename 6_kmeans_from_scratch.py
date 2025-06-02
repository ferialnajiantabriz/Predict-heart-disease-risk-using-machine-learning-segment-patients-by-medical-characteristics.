import numpy as np

def kmeans_from_scratch(data, n_clusters, max_iters=100, tol=1e-4):

    np.random.seed(123)

    if n_clusters > data.shape[0]:
        raise ValueError(
            f"Number of clusters (n_clusters={n_clusters}) cannot exceed the number of data points ({data.shape[0]})."
        )

    random_indices = np.random.choice(data.shape[0], size=n_clusters, replace=False)
    centroids = data[random_indices]

    for iteration in range(max_iters):
        # Compute squared Euclidean distances manually without np.sqrt
        diff = data[:, np.newaxis, :] - centroids
        squared_diff = diff ** 2
        sum_squared_diff = squared_diff.sum(axis=2)

        # Assign clusters based on closest centroid (minimum squared distance)
        cluster_labels = np.argmin(sum_squared_diff, axis=1)

        # Compute new centroids
        new_centroids = np.array([
            data[cluster_labels == i].mean(axis=0) if np.any(cluster_labels == i) else centroids[i]
            for i in range(n_clusters)
        ])

        # Compute SSE (Sum of Squared Errors)
        sse = np.sum([
            np.sum((data[cluster_labels == i] - centroids[i]) ** 2)
            for i in range(n_clusters)
        ])

        # Compute centroid shift without using np.sqrt
        centroid_shift = np.sum((new_centroids - centroids) ** 2)

        # Check for convergence
        if centroid_shift < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids

    return cluster_labels, centroids, sse
