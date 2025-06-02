# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"/Users/ferialnajiantabriz/Desktop/codes/DataMiningPro/heart.csv")

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

# Create copies to avoid SettingWithCopyWarning
X_train = X_train.copy()
X_test = X_test.copy()

X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# Remove outliers from training data only
def remove_outliers(data, columns):
    """
    Removes outliers using the IQR method across specified columns.
    """
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

X_train = remove_outliers(X_train, continuous_cols)
y_train = y_train.loc[X_train.index]  # Match indices of y_train with filtered X_train

# Convert datasets to NumPy arrays
X_train_features = X_train.to_numpy()
y_train_target = y_train.to_numpy()
X_test_features = X_test.to_numpy()
y_test_target = y_test.to_numpy()

# Logistic Regression Implementation from Scratch
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_pred):
        # Log-loss function
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)


# Initialize and train Logistic Regression from scratch
lr_scratch = LogisticRegressionScratch(learning_rate=0.1, max_iter=1000)
lr_scratch.fit(X_train_features, y_train_target)

# Predict on the test set
y_pred = lr_scratch.predict(X_test_features)

# Calculate metrics
accuracy = accuracy_score(y_test_target, y_pred)
precision = precision_score(y_test_target, y_pred)
recall = recall_score(y_test_target, y_pred)
classification_rep = classification_report(y_test_target, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:\n", classification_rep)

# Extract feature importances
lr_feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_scratch.weights
}).sort_values(by='Coefficient', ascending=False)

# Display feature importance
print("\nFeature Importance:\n", lr_feature_importances)

# Optionally, save feature importance to a CSV file
lr_feature_importances.to_csv("logistic_regression_scratch_feature_importance.csv", index=False)

# Generate the confusion matrix
cm = confusion_matrix(y_test_target, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()
