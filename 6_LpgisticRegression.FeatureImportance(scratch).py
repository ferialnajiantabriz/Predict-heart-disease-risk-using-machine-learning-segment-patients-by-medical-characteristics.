import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
np.random.seed(42)
random.seed(42)

# Logistic Regression from Scratch with Enhanced Features
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000, regularization=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.regularization = regularization
        self.lambda_ = lambda_

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_pred):
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        base_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # Add regularization penalty
        if self.regularization == 'l2':
            reg_loss = (self.lambda_ / 2) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            reg_loss = self.lambda_ * np.sum(np.abs(self.weights))
        else:
            reg_loss = 0
        return base_loss + reg_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += (self.lambda_ / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_ / n_samples) * np.sign(self.weights)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            #  Print loss every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {self.loss(y, self.sigmoid(linear_model)):.4f}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

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

# Initialize lists to store accuracies
accuracies_full = []
accuracies_important = []

# Loop over n_runs
for run in range(n_runs):
    print(f"Run {run + 1}/{n_runs}")

    # Split data into training and testing sets
    X_train_full, X_test_full, y_train, y_test = train_test_split(X_values, y, test_size=0.2)

    # First, train using all features


    # Train the Logistic Regression Classifier from scratch with all features
    lr_full = LogisticRegressionScratch(learning_rate=0.01, n_iters=1000, regularization='l2', lambda_=0.01)
    lr_full.fit(X_train_full, y_train)

    # Predict on test set using all features
    y_pred_full = lr_full.predict(X_test_full)

    # Calculate accuracy using all features
    accuracy_full = np.mean(y_pred_full == y_test)
    accuracies_full.append(accuracy_full)

    # select important features and retrain


    # Number of top features to select
    n_top_features = 7

    # Select a random subset of features
    important_features_indices = random.sample(range(len(feature_names)), n_top_features)

    # Select important features from training and test sets
    X_train_important = X_train_full[:, important_features_indices]
    X_test_important = X_test_full[:, important_features_indices]

    # Train the Logistic Regression Classifier from scratch with important features
    lr_important = LogisticRegressionScratch(learning_rate=0.01, n_iters=1000, regularization='l2', lambda_=0.01)
    lr_important.fit(X_train_important, y_train)

    # Predict on test set using important features
    y_pred_important = lr_important.predict(X_test_important)

    # Calculate accuracy using important features
    accuracy_important = np.mean(y_pred_important == y_test)
    accuracies_important.append(accuracy_important)
    # Extract weights as feature importance for the full-feature model
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_full.weights
    }).sort_values(by='Coefficient', ascending=False)

    # Display the top 7 features
    print("\nTop Features based on Coefficients:")
    print(feature_importances.head(7))

# After the loop, compute the mean and std of accuracies
mean_accuracy_full = np.mean(accuracies_full)
std_accuracy_full = np.std(accuracies_full)

mean_accuracy_important = np.mean(accuracies_important)
std_accuracy_important = np.std(accuracies_important)

print(f"\nAccuracy with all features over {n_runs} runs: {mean_accuracy_full:.4f} ± {std_accuracy_full:.4f}")
print(f"Accuracy with selected important features over {n_runs} runs: {mean_accuracy_important:.4f} ± {std_accuracy_important:.4f}")
