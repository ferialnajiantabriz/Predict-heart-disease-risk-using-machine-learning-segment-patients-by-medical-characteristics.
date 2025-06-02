import numpy as np
from collections import Counter

# A custom implementation of a decision tree classifier, including feature importance calculation.
class DecisionTreeClassifierScratchWithImportance:
    def __init__(self, max_depth=None):
        # Initialize the classifier with a maximum depth for the tree .
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None

    def entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        # Compute entropy using the formula: -sum(p * log2(p))
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def information_gain(self, X_column, y, threshold):
        # Calculate the information gain from splitting on a given threshold.
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        parent_entropy = self.entropy(y)
        n = len(y)  # Total number of samples.
        n_left, n_right = sum(left_indices), sum(right_indices)

        # Avoid division by zero if a split results in empty groups.
        if n_left == 0 or n_right == 0:
            return 0

        # Calculate entropy for the left and right splits.
        e_left = self.entropy(y[left_indices])
        e_right = self.entropy(y[right_indices])
        # Compute weighted average of child entropies.
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        # Information gain is the reduction in entropy.
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        # Find the best feature and threshold to split the data.
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            X_column = X[:, feature_index]  # Extract column for the current feature.
            thresholds = np.unique(X_column)  # Unique thresholds to evaluate.
            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)
                if gain > best_gain:  # Update best split if gain is higher.
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, n_samples_total, depth=0, feature_gains=None):
        # Recursively build the decision tree.
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria: max depth reached, pure node, or insufficient samples.
        if depth == self.max_depth or num_labels == 1 or n_samples < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value

        # Find the best split for the current node.
        feature, threshold, gain = self.best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value

        # Update feature importances using weighted information gain.
        if feature_gains is not None:
            weighted_gain = (n_samples / n_samples_total) * gain
            feature_gains[feature] += weighted_gain

        # Split the data based on the best threshold.
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        # Recursively build left and right subtrees.
        left_subtree = self.build_tree(
            X[left_indices], y[left_indices], n_samples_total, depth + 1, feature_gains
        )
        right_subtree = self.build_tree(
            X[right_indices], y[right_indices], n_samples_total, depth + 1, feature_gains
        )
        # Return the node
        return {"feature": feature, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        # Fit the decision tree to the data `X` and labels `y`.
        n_features = X.shape[1]
        feature_gains = np.zeros(n_features)
        n_samples_total = len(y)
        # Build the tree and calculate feature importances.
        self.tree = self.build_tree(X, y, n_samples_total, feature_gains=feature_gains)
        total_gain = np.sum(feature_gains)
        if total_gain > 0:  # Normalize feature importances if any gain is found.
            self.feature_importances_ = feature_gains / total_gain
        else:  # If no splits were made, importances remain zero.
            self.feature_importances_ = np.zeros_like(feature_gains)

    def predict_one(self, x, tree):
        # Predict the class for a single data point `x` using the tree.
        if not isinstance(tree, dict):
            return tree
        feature = tree["feature"]
        threshold = tree["threshold"]
        # Recurse to the left or right subtree based on the feature value.
        if x[feature] <= threshold:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    def predict(self, X):
        # Predict the class for each data point
        return np.array([self.predict_one(x, self.tree) for x in X])
