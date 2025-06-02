import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
np.random.seed(42)
random.seed(42)
# Decision Tree Implementation from Scratch with Feature Importance
class DecisionTreeClassifierScratchWithImportance:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None

    def entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def information_gain(self, X_column, y, threshold):
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = sum(left_indices), sum(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        e_left = self.entropy(y[left_indices])
        e_right = self.entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return parent_entropy - child_entropy

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, n_samples_total, depth=0, feature_gains=None):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))
        if depth == self.max_depth or num_labels == 1 or n_samples < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value
        feature, threshold, gain = self.best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value

        # Update feature importance with sample weighting
        if feature_gains is not None:
            weighted_gain = (n_samples / n_samples_total) * gain
            feature_gains[feature] += weighted_gain

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_subtree = self.build_tree(
            X[left_indices], y[left_indices], n_samples_total, depth + 1, feature_gains
        )
        right_subtree = self.build_tree(
            X[right_indices], y[right_indices], n_samples_total, depth + 1, feature_gains
        )
        return {"feature": feature, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        n_features = X.shape[1]
        feature_gains = np.zeros(n_features)
        n_samples_total = len(y)
        self.tree = self.build_tree(X, y, n_samples_total, feature_gains=feature_gains)
        total_gain = np.sum(feature_gains)
        if total_gain > 0:
            self.feature_importances_ = feature_gains / total_gain
        else:
            self.feature_importances_ = np.zeros_like(feature_gains)

    def predict_one(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

# Load the dataset
# Load the dataset
df = pd.read_csv(r"/Users/ferialnajiantabriz/Desktop/codes/DataMiningPro/pheart.csv")

# Drop unnecessary columns if they exist
df = df.drop(columns=["Unnamed: 0", "Unnamed: 1"], errors="ignore")

# Identify categorical and numerical features
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = df.drop(columns=categorical_cols + ['target']).columns.tolist()

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
#Removes the target column from the dataset
X = df.drop(columns=["target"])
y = df["target"].values

# Convert DataFrame to NumPy array
X_values = X.values
feature_names = X.columns.tolist()

# Number of runs
n_runs = 10

# Number of bootstraps within each run(random resampling iterations)
n_bootstraps = 10  # Adjusted to reduce computation time

# Initialize lists to store accuracies
accuracies_full = []
accuracies_important = []

# Loop over n_runs
for run in range(n_runs):
    print(f"Run {run + 1}/{n_runs}")

    # Split data into training and testing sets
    X_train_full, X_test_full, y_train, y_test = train_test_split(X_values, y, test_size=0.2)


    # Bootstrapping to Calculate Mean and Std of Feature Importances


    n_features = X_train_full.shape[1]
    importances_bootstrap = np.zeros((n_bootstraps, n_features))

    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(len(X_train_full), size=len(X_train_full), replace=True)
        X_bootstrap = X_train_full[indices]
        y_bootstrap = y_train[indices]

        # Train model
        dt_bootstrap = DecisionTreeClassifierScratchWithImportance(max_depth=5)
        dt_bootstrap.fit(X_bootstrap, y_bootstrap)

        # Store importances
        importances_bootstrap[i] = dt_bootstrap.feature_importances_

    # Calculate mean and standard deviation
    importances_mean = np.mean(importances_bootstrap, axis=0)
    importances_std = np.std(importances_bootstrap, axis=0)

    # Combine feature names, mean importances, and standard deviations into a list of tuples
    feature_importances_with_confidence = [
        (feature_names[i], importances_mean[i], importances_std[i]) for i in range(len(feature_names))
    ]

    # Sort the list by mean importance in descending order
    feature_importances_with_confidence.sort(key=lambda x: x[1], reverse=True)

    # Display feature importances
    print("\nFeature Importances with Confidence Intervals:")
    for feature, mean, std in feature_importances_with_confidence:
        print(f"Feature: {feature}, Importance: {mean:.4f}, Std Dev: {std:.4f}")


    # First, train using all features


    # Train the Decision Tree Classifier from scratch with all features
    dt_scratch_full = DecisionTreeClassifierScratchWithImportance(max_depth=5)
    dt_scratch_full.fit(X_train_full, y_train)

    # Predict on test set using all features
    y_pred_full = dt_scratch_full.predict(X_test_full)

    # Calculate accuracy using all features
    accuracy_full = np.mean(y_pred_full == y_test)
    accuracies_full.append(accuracy_full)



    # Number of top features to select
    n_top_features = 7

    # Get indices of features sorted by importance (descending order)
    sorted_indices = np.argsort(-importances_mean)

    # Select the top n feature indices
    important_features_indices = sorted_indices[:n_top_features]

    # If fewer than n features exist , use all features
    if len(important_features_indices) == 0:
        important_features_indices = range(len(feature_names))

    # Select important features from training and test sets
    X_train_important = X_train_full[:, important_features_indices]
    X_test_important = X_test_full[:, important_features_indices]

    # Train the Decision Tree Classifier from scratch with important features
    dt_scratch_important = DecisionTreeClassifierScratchWithImportance(max_depth=7)
    dt_scratch_important.fit(X_train_important, y_train)

    # Predict on test set using important features
    y_pred_important = dt_scratch_important.predict(X_test_important)

    # Calculate accuracy using important features
    accuracy_important = np.mean(y_pred_important == y_test)
    accuracies_important.append(accuracy_important)

# After the loop, compute the mean and std of accuracies
mean_accuracy_full = np.mean(accuracies_full)
std_accuracy_full = np.std(accuracies_full)

mean_accuracy_important = np.mean(accuracies_important)
std_accuracy_important = np.std(accuracies_important)

print(f"\nAccuracy with all features over {n_runs} runs: {mean_accuracy_full:.4f} ± {std_accuracy_full:.4f}")
print(
    f"Accuracy with selected important features over {n_runs} runs: {mean_accuracy_important:.4f} ± {std_accuracy_important:.4f}")



import matplotlib.pyplot as plt

# Define the accuracy metrics
accuracy_metrics = {
    "All Features": (mean_accuracy_full, std_accuracy_full),
    "Important Features": (mean_accuracy_important, std_accuracy_important),
}

# Extract data for plotting
categories = list(accuracy_metrics.keys())
means = [accuracy_metrics[cat][0] for cat in categories]
std_devs = [accuracy_metrics[cat][1] for cat in categories]

# Plotting
plt.figure(figsize=(8, 6))
bar_width = 0.5
colors = ['skyblue', 'salmon']

bars = plt.bar(categories, means, yerr=std_devs, capsize=5, color=colors, edgecolor="black", width=bar_width)

# Add accuracy values as text labels above the bars
for bar, mean, std in zip(bars, means, std_devs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{mean:.4f} ± {std:.4f}', ha='center', va='bottom', fontsize=10)

# Formatting the plot
plt.title("Accuracy Comparison for Decision Tree Classifier", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Feature Set", fontsize=12)
plt.ylim(0.8, 1.0)  # Adjust the y-axis range for better visibility
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
