# Import required libraries
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# load the dataset
df = pd.read_csv("/Users/ferialnajiantabriz/Desktop/codes/DataMiningPro/heart.csv")

# Encoding Categorical Variables
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split the dataset into features (X) and target (y)
X = df.drop(columns=["target"])
y = df["target"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Continuous Variables
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# Remove outliers from training data only
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for col in continuous_cols:
    X_train = remove_outliers(X_train, col)

# Match y_train indices with X_train
y_train = y_train.loc[X_train.index]

# Convert datasets to NumPy arrays for custom classifiers
X_train_features = X_train.to_numpy()
y_train_target = y_train.to_numpy()
X_test_features = X_test.to_numpy()
y_test_target = y_test.to_numpy()

# Initialize the accuracies dictionary
accuracies = {}

# KNN Classifier from Scratch
class KNNClassifierScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))


# Logistic Regression from Scratch
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


# Naive Bayes from Scratch
class NaiveBayesScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)

    def _predict_one(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(
                np.log(self._gaussian_density(x, self.mean[c], self.var[c]))
            )
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _gaussian_density(self, x, mean, var):
        eps = 1e-6  # Small constant to avoid division by zero
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent
class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

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
        for feature_index in range(X.shape[1]):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))
        if depth == self.max_depth or num_labels == 1 or n_samples < 2:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value
        feature, threshold = self.best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return leaf_value
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        return {"feature": feature, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

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

# Use the custom Decision Tree
dt_scratch = DecisionTreeClassifierScratch(max_depth=5)
dt_scratch.fit(X_train_features, y_train_target)

# Predict and evaluate accuracy
y_pred = dt_scratch.predict(X_test_features)
dt_scratch_accuracy = accuracy_score(y_test_target, y_pred)

# Add accuracy to the results
accuracies["Decision Tree (Scratch) Accuracy"] = dt_scratch_accuracy

# KNN from Scratch
knn_scratch = KNNClassifierScratch(k=5)
knn_scratch.fit(X_train_features, y_train_target)
y_pred_knn = knn_scratch.predict(X_test_features)
knn_scratch_accuracy = accuracy_score(y_test_target, y_pred_knn)
accuracies["KNN (Scratch) Accuracy"] = knn_scratch_accuracy

# Logistic Regression from Scratch
logistic_reg_scratch = LogisticRegressionScratch(learning_rate=0.01, n_iters=1000)
logistic_reg_scratch.fit(X_train_features, y_train_target)
y_pred_logistic_scratch = logistic_reg_scratch.predict(X_test_features)
logistic_reg_scratch_accuracy = accuracy_score(y_test_target, y_pred_logistic_scratch)
accuracies["Logistic Regression (Scratch) Accuracy"] = logistic_reg_scratch_accuracy

# Naive Bayes from Scratch
naive_bayes_scratch = NaiveBayesScratch()
naive_bayes_scratch.fit(X_train_features, y_train_target)
y_pred_naive_bayes = naive_bayes_scratch.predict(X_test_features)
naive_bayes_scratch_accuracy = accuracy_score(y_test_target, y_pred_naive_bayes)
accuracies["Naive Bayes (Scratch) Accuracy"] = naive_bayes_scratch_accuracy

# Print final accuracies
#print(accuracies)
# Initialize the accuracies dictionary
accuracies = {}

# Library-Based Classifiers

# Decision Tree with library
dt_library = DecisionTreeClassifier(random_state=42)
dt_library.fit(X_train_features, y_train_target)
y_pred_dt_library = dt_library.predict(X_test_features)
dt_library_accuracy = accuracy_score(y_test_target, y_pred_dt_library)
accuracies["Decision Tree (Library) Accuracy"] = dt_library_accuracy

# KNN with library
knn_library = KNeighborsClassifier(n_neighbors=5)
knn_library.fit(X_train_features, y_train_target)
y_pred_knn_library = knn_library.predict(X_test_features)
knn_library_accuracy = accuracy_score(y_test_target, y_pred_knn_library)
accuracies["KNN (Library) Accuracy"] = knn_library_accuracy

# Logistic Regression with library
lr_library = LogisticRegression(max_iter=1000, random_state=42)
lr_library.fit(X_train_features, y_train_target)
y_pred_lr_library = lr_library.predict(X_test_features)
lr_library_accuracy = accuracy_score(y_test_target, y_pred_lr_library)
accuracies["Logistic Regression (Library) Accuracy"] = lr_library_accuracy

# Naive Bayes with library
nb_library = GaussianNB()
nb_library.fit(X_train_features, y_train_target)
y_pred_nb_library = nb_library.predict(X_test_features)
nb_library_accuracy = accuracy_score(y_test_target, y_pred_nb_library)
accuracies["Naive Bayes (Library) Accuracy"] = nb_library_accuracy

# Print accuracies of library-based models
# print("Library-Based Models Accuracies:")
# print(accuracies)
import numpy as np
from collections import defaultdict

# Number of iterations
n_iterations = 5

# Initialize dictionaries to store accuracies for each model
all_accuracies = defaultdict(list)

for _ in range(n_iterations):
    # Shuffle and split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Normalize Continuous Variables
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # Remove outliers from training data only
    for col in continuous_cols:
        X_train = remove_outliers(X_train, col)
    y_train = y_train.loc[X_train.index]

    # Convert datasets to NumPy arrays for scratch implementations
    X_train_features = X_train.to_numpy()
    y_train_target = y_train.to_numpy()
    X_test_features = X_test.to_numpy()
    y_test_target = y_test.to_numpy()

    # Library-Based Models
    dt_library = DecisionTreeClassifier(random_state=42)
    dt_library.fit(X_train_features, y_train_target)
    all_accuracies["Decision Tree (Library)"].append(accuracy_score(y_test_target, dt_library.predict(X_test_features)))

    knn_library = KNeighborsClassifier(n_neighbors=5)
    knn_library.fit(X_train_features, y_train_target)
    all_accuracies["KNN (Library)"].append(accuracy_score(y_test_target, knn_library.predict(X_test_features)))

    lr_library = LogisticRegression(max_iter=1000, random_state=42)
    lr_library.fit(X_train_features, y_train_target)
    all_accuracies["Logistic Regression (Library)"].append(accuracy_score(y_test_target, lr_library.predict(X_test_features)))

    nb_library = GaussianNB()
    nb_library.fit(X_train_features, y_train_target)
    all_accuracies["Naive Bayes (Library)"].append(accuracy_score(y_test_target, nb_library.predict(X_test_features)))

    # Scratch Implementations
    dt_scratch = DecisionTreeClassifierScratch(max_depth=5)
    dt_scratch.fit(X_train_features, y_train_target)
    all_accuracies["Decision Tree (Scratch)"].append(accuracy_score(y_test_target, dt_scratch.predict(X_test_features)))

    knn_scratch = KNNClassifierScratch(k=5)
    knn_scratch.fit(X_train_features, y_train_target)
    all_accuracies["KNN (Scratch)"].append(accuracy_score(y_test_target, knn_scratch.predict(X_test_features)))

    logistic_reg_scratch = LogisticRegressionScratch(learning_rate=0.01, n_iters=1000)
    logistic_reg_scratch.fit(X_train_features, y_train_target)
    all_accuracies["Logistic Regression (Scratch)"].append(accuracy_score(y_test_target, logistic_reg_scratch.predict(X_test_features)))

    naive_bayes_scratch = NaiveBayesScratch()
    naive_bayes_scratch.fit(X_train_features, y_train_target)
    all_accuracies["Naive Bayes (Scratch)"].append(accuracy_score(y_test_target, naive_bayes_scratch.predict(X_test_features)))

# Calculate mean and standard deviation for each model
final_results = {model: {"Mean Accuracy": np.mean(acc), "Std Dev": np.std(acc)} for model, acc in all_accuracies.items()}

# Print final results
for model, stats in final_results.items():
    print(f"{model}: Mean Accuracy = {stats['Mean Accuracy']:.4f}, Std Dev = {stats['Std Dev']:.4f}")

    import matplotlib.pyplot as plt
    import numpy as np

    # Define the accuracy data
    models = [
        "Decision Tree (Library)", "KNN (Library)", "Logistic Regression (Library)", "Naive Bayes (Library)",
        "Decision Tree (Scratch)", "KNN (Scratch)", "Logistic Regression (Scratch)", "Naive Bayes (Scratch)"
    ]

    mean_accuracies = [
        0.9776, 0.8166, 0.8615, 0.8312,  # Library models
        0.9034, 0.8166, 0.8332, 0.8312  # Scratch implementations
    ]

    std_devs = [
        0.0073, 0.0173, 0.0050, 0.0165,  # Library models
        0.0139, 0.0173, 0.0189, 0.0165  # Scratch implementations
    ]

    # Plot settings
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(models))
    colors = ['skyblue' if "Library" in model else 'salmon' for model in models]

    # Create the bar plot
    bars = plt.bar(x_pos, mean_accuracies, yerr=std_devs, capsize=5, color=colors, edgecolor="black")

    # Add values above the bars
    for bar, mean, std in zip(bars, mean_accuracies, std_devs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{mean:.4f} ± {std:.4f}', ha='center', va='bottom', fontsize=10)

    # Formatting the plot
    plt.xticks(x_pos, models, rotation=45, ha='right', fontsize=10)
    plt.title("Model Accuracy Comparison (Library vs. Scratch Implementations)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.75, 1.0)  # Adjust y-axis range for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()
