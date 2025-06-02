import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)


class KNNClassifierScratch:
    def __init__(self, k=5):

        self.k = k

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict_one(self, x):

        # Compute distances to all training samples
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Find the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common label
        return Counter(k_nearest_labels).most_common(1)[0][0]

    def predict(self, X):

        return np.array([self.predict_one(x) for x in X])


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
X = df.drop(columns=["target"])
y = df["target"].values

# Convert DataFrame to NumPy array
X_values = X.values
feature_names = X.columns.tolist()

# Number of runs
n_runs = 10

# Number of bootstraps within each run
n_bootstraps = 10

# Initialize lists to store accuracies
accuracies_full = []
accuracies_important = []

# Initialize feature importance tracker
average_importances = np.zeros(X_values.shape[1])

# Loop over n_runs
for run in range(n_runs):
    print(f"Run {run + 1}/{n_runs}")

    # Split data into training and testing sets
    X_train_full, X_test_full, y_train, y_test = train_test_split(X_values, y, test_size=0.2)

    # Feature Importance using Bootstrapping

    n_features = X_train_full.shape[1]
    feature_importances = np.zeros(n_features)

    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(len(X_train_full), size=len(X_train_full), replace=True)
        X_bootstrap = X_train_full[indices]
        y_bootstrap = y_train[indices]

        for feature_index in range(n_features):
            # Compute importance of each feature via its effect on distance
            X_temp = np.delete(X_bootstrap, feature_index, axis=1)
            knn = KNNClassifierScratch(k=5)
            knn.fit(X_temp, y_bootstrap)
            distances = np.sqrt(np.sum((X_temp - X_bootstrap[:, :-1]) ** 2, axis=1))
            feature_importances[feature_index] += np.mean(distances)

    # Normalize feature importances
    feature_importances /= np.sum(feature_importances)
    average_importances += feature_importances

    # Sort features by importance
    sorted_indices = np.argsort(-feature_importances)
    n_top_features = 7
    important_features_indices = sorted_indices[:n_top_features]

    # Classification with All Features
    knn_full = KNNClassifierScratch(k=5)
    knn_full.fit(X_train_full, y_train)
    y_pred_full = knn_full.predict(X_test_full)
    accuracy_full = np.mean(y_pred_full == y_test)
    accuracies_full.append(accuracy_full)

    # Classification with Important Features
    X_train_important = X_train_full[:, important_features_indices]
    X_test_important = X_test_full[:, important_features_indices]

    knn_important = KNNClassifierScratch(k=5)
    knn_important.fit(X_train_important, y_train)
    y_pred_important = knn_important.predict(X_test_important)
    accuracy_important = np.mean(y_pred_important == y_test)
    accuracies_important.append(accuracy_important)

# Average the feature importances over runs
average_importances /= n_runs

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": average_importances
})

# Sort features by importance in descending order
feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)

# Display feature importances
print("\nFeature Importances (Sorted):")
print(feature_importances_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df["Feature"], feature_importances_df["Importance"], color="blue")
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("KNN Feature Importances", fontsize=14)
plt.gca().invert_yaxis()  # Invert y-axis for descending order
plt.tight_layout()
plt.show()

# Calculate accuracies
mean_accuracy_full = np.mean(accuracies_full)
std_accuracy_full = np.std(accuracies_full)

mean_accuracy_important = np.mean(accuracies_important)
std_accuracy_important = np.std(accuracies_important)

print(f"\nAccuracy with all features over all runs for KNN: {mean_accuracy_full:.4f} ± {std_accuracy_full:.4f}")
print(
    f"Accuracy with selected important features over all runs with KNN: {mean_accuracy_important:.4f} ± {std_accuracy_important:.4f}")

# Correct data for accuracies and errors
labels = ['All Features', 'Important Features']
means = [0.8337, 0.8264]  # Correct mean accuracies
errors = [0.0190, 0.0247]  # Correct standard deviations

# Create the bar plot
plt.figure(figsize=(8, 6))
colors = ['cornflowerblue', 'lightcoral']
bars = plt.bar(labels, means, yerr=errors, color=colors, capsize=8, edgecolor='black', width=0.4)  # Adjust width here

# Adding accuracy values above bars
for bar, mean, error in zip(bars, means, errors):
    plt.text(bar.get_x() + bar.get_width() / 2, mean + error + 0.01,
             f'{mean:.4f} ± {error:.4f}', ha='center', fontsize=12, fontweight='bold')

# Improve plot aesthetics
plt.title('Comparison of KNN Accuracy with All vs. Important Features', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Feature Set', fontsize=12)
plt.ylim(0, 1.05)  # Extend y-axis slightly above 1 for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
