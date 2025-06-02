# app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from scipy import stats
from collections import Counter

# Import Kmeans scratch implementations
from kmeans_from_scratch import kmeans_from_scratch

# Set random seed for reproducibility
np.random.seed(42)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Heart Disease Prediction and Patient Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main title and description
st.title("Heart Disease Prediction and Patient Segmentation")

st.markdown("""
This application provides two main functionalities:

1. **Heart Disease Risk Prediction**: Predict whether a person is at high or low risk for heart disease based on health metrics using either a Decision Tree or Logistic Regression algorithm implemented from scratch.

2. **Patient Segmentation**: Perform K-Means clustering to segment patients into distinct groups using your custom K-Means algorithm and provide tailored health advice based on cluster assignments.
""")

# Sidebar navigation
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Heart Disease Prediction", "Patient Segmentation"]
)

# Logistic Regression from Scratch
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000, regularization=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.regularization = regularization  # 'l1', 'l2', or None
        self.lambda_ = lambda_  # Regularization strength

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

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.array([1 if i >= 0.5 else 0 for i in y_pred])

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.vstack([1 - y_pred, y_pred]).T

# Decision Tree Implementation from Scratch with Feature Importance
class DecisionTreeClassifierScratchWithImportance:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None
        self.classes_ = None

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
            leaf_value = self.create_leaf(y)
            return leaf_value
        feature, threshold, gain = self.best_split(X, y)
        if feature is None:
            leaf_value = self.create_leaf(y)
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

    def create_leaf(self, y):
        counts = np.zeros(len(self.classes_))
        for idx, c in enumerate(self.classes_):
            counts[idx] = np.sum(y == c)
        probabilities = counts / counts.sum()
        predicted_class = self.classes_[np.argmax(probabilities)]
        return {"leaf": True, "probabilities": probabilities, "predicted_class": predicted_class}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
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
        if "leaf" in tree:
            # Return the predicted class
            return tree["predicted_class"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    def predict_proba_one(self, x, tree):
        if "leaf" in tree:
            # Return the probabilities
            return tree["probabilities"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self.predict_proba_one(x, tree["left"])
        else:
            return self.predict_proba_one(x, tree["right"])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

    def predict_proba(self, X):
        return np.array([self.predict_proba_one(x, self.tree) for x in X])

# Heart Disease Prediction

def heart_disease_prediction():
    st.header("Heart Disease Risk Prediction")
    st.write("Predict whether a person is at high or low risk for heart disease.")

    # Load data or manual input
    st.subheader("Input Data")
    top_features = ["cp", "thal", "ca", "oldpeak", "age", "thalach", "chol"]

    # Allow CSV upload or manual input
    data_source = st.radio("Choose input method:", ["Upload CSV", "Manual Input"])

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with the required features:", type="csv")
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview:")
            st.dataframe(input_data)

            # Check if required columns are present
            missing_features = [feature for feature in top_features if feature not in input_data.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                return

            # Prepare data for prediction
            X = input_data[top_features]
            y_true = input_data['target'].values if 'target' in input_data.columns else None
    else:
        # Manual Input
        manual_data = {feature: st.number_input(f"Enter {feature}:", value=0.0) for feature in top_features}
        X = pd.DataFrame([manual_data])
        y_true = None  # No true label for manual input

    # Select the prediction model
    st.subheader("Select Prediction Model")
    model_choice = st.selectbox("Choose the prediction model:", ["Decision Tree", "Logistic Regression"])

    # Train or load the selected model
    st.subheader("Train and Predict")
    train_button = st.button("Train Model and Predict")
    if train_button:
        # Load heart disease dataset for training
        df = pd.read_csv("pheart.csv")
        df = df.drop(columns=["Unnamed: 0", "Unnamed: 1"], errors="ignore")
        X_train = df[top_features]
        y_train = df["target"].values

        # Encode categorical features if necessary
        categorical_features = ['cp', 'thal', 'ca']
        le_dict = {}
        for feature in categorical_features:
            le = LabelEncoder()
            X_train[feature] = le.fit_transform(X_train[feature])
            if feature in X.columns:
                X[feature] = le.transform(X[feature])
            le_dict[feature] = le

        # Convert DataFrame to NumPy array after encoding
        X_train = X_train.values.astype(float)
        X = X.values.astype(float)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X = scaler.transform(X)

        if model_choice == "Decision Tree":
            # Train Decision Tree from scratch
            model = DecisionTreeClassifierScratchWithImportance(max_depth=5)
            model.fit(X_train, y_train)
            st.success("Model trained using Decision Tree from scratch.")
        elif model_choice == "Logistic Regression":
            # Train Logistic Regression from scratch
            model = LogisticRegressionScratch(learning_rate=0.01, n_iters=1000, regularization='l2', lambda_=0.01)
            model.fit(X_train, y_train)
            st.success("Model trained using Logistic Regression.")

        # Make predictions and measure prediction time
        st.subheader("Prediction Results")
        start_time = time.time()
        predictions = model.predict(X)
        end_time = time.time()
        prediction_time = end_time - start_time

        risks = ["High Risk" if pred == 1 else "Low Risk" for pred in predictions]
        st.write("### Predictions:")
        for i, risk in enumerate(risks):
            st.write(f"Sample {i + 1}: **{risk}**")

        st.write(f"Prediction Time: {prediction_time:.4f} seconds")

        # If true labels are available, compute metrics
        if y_true is not None:
            st.subheader("Model Performance Metrics")
            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(y_true, predictions, zero_division=0)
            recall = recall_score(y_true, predictions, zero_division=0)
            f1 = f1_score(y_true, predictions, zero_division=0)
            cm = confusion_matrix(y_true, predictions)

            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1 Score: {f1:.4f}")

            # Display confusion matrix
            st.write("### Confusion Matrix:")
            st.write(cm)

            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # Classification report
            st.write("### Classification Report:")
            report = classification_report(y_true, predictions, zero_division=0)
            st.text(report)

            # Statistical significance test (Chi-squared test)
            st.subheader("Statistical Significance Test")
            try:
                chi2, p, dof, ex = stats.chi2_contingency(cm)
                st.write(f"Chi-squared Statistic: {chi2:.4f}")
                st.write(f"P-value: {p:.4f}")

                if p < 0.05:
                    st.write("The result is statistically significant (p < 0.05).")
                else:
                    st.write("The result is not statistically significant (p >= 0.05).")
            except ValueError as e:
                st.error(f"Chi-squared test could not be performed: {e}")

            # ROC Curve
            st.write("### ROC Curve:")
            try:
                y_score = model.predict_proba(X)[:, np.where(model.classes_ == 1)[0][0]]
                fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
                auc = roc_auc_score(y_true, y_score)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax_roc.legend(loc='lower right')
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            except Exception as e:
                st.write("ROC Curve could not be generated.")
                st.write(f"Error: {e}")

        else:
            st.info("True labels are not provided. Performance metrics cannot be calculated.")

# Patient Segmentation
def patient_segmentation():
    st.header("Patient Segmentation")
    st.write("Segment patients into clusters using the custom K-Means algorithm and provide tailored advice.")

    # Load data
    st.subheader("Input Data")
    uploaded_file = st.file_uploader("Upload a CSV file with patient data:", type="csv")
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
        return

    # Read and preprocess data
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df)

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Ensure required columns are present
    required_columns = categorical_cols + continuous_cols
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Normalize continuous variables
    scaler = StandardScaler()
    df_encoded[continuous_cols] = scaler.fit_transform(df_encoded[continuous_cols])

    # Remove outliers from continuous variables
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

    # Remove outliers and keep the same indices
    df_encoded = remove_outliers(df_encoded, continuous_cols)

    # Synchronize df with df_encoded
    df = df.loc[df_encoded.index]

    # Reset indices to ensure alignment
    df = df.reset_index(drop=True)
    df_encoded = df_encoded.reset_index(drop=True)

    st.success("Data preprocessed successfully.")

    # Convert the preprocessed data to a NumPy array for clustering
    X_features = df_encoded.values

    # Elbow Method
    st.subheader("Determine Optimal Number of Clusters (Elbow Method)")
    compute_elbow = st.button("Compute Elbow Method")
    if compute_elbow:
        k_values = range(2, 11)
        total_sse = []
        st.write("Calculating SSE for different values of K:")
        progress_bar = st.progress(0)
        for idx, k in enumerate(k_values):
            cluster_labels, centroids, sse = kmeans_from_scratch(X_features, n_clusters=k)
            total_sse.append(sse)
            progress_bar.progress((idx + 1) / len(k_values))
            st.write(f"K = {k}, SSE = {sse}")

        # Plot the Elbow Method
        fig_elbow = plt.figure(figsize=(10, 6))
        plt.plot(list(k_values), total_sse, marker="o", linestyle="--")
        plt.axvline(x=4, color="red", linestyle="--", label="Suggested K=4")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Total Sum of Squared Errors (SSE)")
        plt.title("Elbow Method for Optimal K")
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

    # Select number of clusters
    st.subheader("K-Means Clustering")
    k = st.slider("Select the number of clusters (K):", min_value=2, max_value=10, value=4)
    run_clustering = st.button("Run Clustering")

    if run_clustering:
        start_time = time.time()
        cluster_labels, centroids, sse = kmeans_from_scratch(X_features, n_clusters=k)
        end_time = time.time()
        clustering_time = end_time - start_time
        st.write(f"Clustering completed in {clustering_time:.2f} seconds.")

        # Add cluster labels to the original dataset
        df["Cluster"] = cluster_labels

        # Define advice for each cluster
        cluster_advice = {
            0: "Cluster 0: Patients may benefit from increased physical activity and a balanced diet.",
            1: "Cluster 1: Patients should monitor cholesterol levels and consult a healthcare provider regularly.",
            2: "Cluster 2: Patients are advised to manage stress and maintain regular check-ups.",
            3: "Cluster 3: Tailored interventions are recommended due to diverse risk factors.",

        }

        # Handle cases where k exceeds the defined advice
        for i in range(k):
            if i not in cluster_advice:
                cluster_advice[i] = f"Cluster {i}: General advice to maintain a healthy lifestyle."

        # Map cluster labels to advice
        df["Advice"] = df["Cluster"].map(cluster_advice)

        # Display the dataframe with advice
        st.write("### Clustered Data with Advice:")
        st.dataframe(df)

        # Silhouette Score and Silhouette Plot
        st.subheader("Silhouette Analysis")
        silhouette_avg = silhouette_score(X_features, cluster_labels)
        st.write(f"Average Silhouette Score for K={k}: {silhouette_avg:.4f}")

        # Silhouette Plot
        st.write("### Silhouette Plot:")
        silhouette_vals = silhouette_samples(X_features, cluster_labels)
        fig_silhouette = plt.figure(figsize=(10, 6))
        y_lower = 10
        for i in range(k):
            ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
            ith_cluster_silhouette_vals.sort()
            size_cluster_i = ith_cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.nipy_spectral(float(i) / k)
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals,
                              facecolor=color, edgecolor=color, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
            y_lower = y_upper + 10

        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.title("Silhouette Plot for K-Means Clustering")
        plt.yticks([])
        plt.tight_layout()
        st.pyplot(fig_silhouette)
        plt.close(fig_silhouette)

        # Pairwise Relationships for Clusters
        st.subheader("Pairwise Relationships for Clusters")
        X_visual = df_encoded.copy()
        X_visual['Cluster'] = cluster_labels
        selected_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        with st.spinner("Generating pairplot..."):
            fig_pairplot = sns.pairplot(X_visual, hue='Cluster', vars=selected_vars, palette='Set2', plot_kws={'alpha': 0.7})
            fig_pairplot.fig.suptitle(f'Pairwise Relationships for Clusters (K={k})', y=1.02)
            st.pyplot(fig_pairplot)
            plt.close()

        # Bar plot for the number of data points in each cluster
        st.subheader("Number of Data Points in Each Cluster")
        cluster_sizes = X_visual['Cluster'].value_counts().sort_index()

        fig_cluster_sizes = plt.figure(figsize=(8, 6))
        cluster_sizes.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title("Number of Data Points in Each Cluster", fontsize=14)
        plt.xlabel("Cluster Label", fontsize=12)
        plt.ylabel("Number of Data Points", fontsize=12)
        plt.xticks(rotation=0, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_cluster_sizes)
        plt.close(fig_cluster_sizes)

        # PCA Scatter Plot
        st.subheader("PCA Visualization of Clusters")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_features)
        X_visual['PCA1'] = pca_result[:, 0]
        X_visual['PCA2'] = pca_result[:, 1]

        fig_pca = plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='Cluster', data=X_visual, palette='Set2', s=100, alpha=0.7, edgecolor="k"
        )
        plt.title("Scatter Plot of Clusters (K-Means Clustering)", fontsize=14)
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.legend(title="Cluster", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_pca)
        plt.close(fig_pca)

        # Bar plots for categorical features
        st.subheader("Cluster Distribution for Categorical Features")
        one_hot_categorical_cols = [col for col in df_encoded.columns if any(col.startswith(prefix + '_') for prefix in categorical_cols)]

        for col in one_hot_categorical_cols:
            fig_bar = plt.figure(figsize=(8, 6))
            sns.barplot(data=X_visual, x='Cluster', y=col, ci=None, palette='Set2')
            plt.title(f'Cluster Distribution for {col}')
            plt.xlabel('Cluster')
            plt.ylabel('Proportion')
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        # Analyze continuous variables (mean, standard deviation) for each cluster
        st.subheader("Cluster Analysis (Continuous Variables)")
        cluster_analysis = X_visual.groupby('Cluster').agg({
            'age': ['mean', 'std'],
            'trestbps': ['mean', 'std'],
            'chol': ['mean', 'std'],
            'thalach': ['mean', 'std'],
            'oldpeak': ['mean', 'std']
        })
        st.write(cluster_analysis)

        # Analyze categorical features
        st.subheader("Cluster Analysis (Categorical Variables)")
        categorical_analysis = X_visual.groupby('Cluster')[one_hot_categorical_cols].mean()
        st.write(categorical_analysis)

        # Display cluster-specific advice
        st.subheader("Cluster-Specific Advice")
        for cluster_id, advice in cluster_advice.items():
            st.write(f"**Cluster {cluster_id}:** {advice}")

        # Save segmented dataset with advice to a CSV file
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Segmented Data with Advice",
            data=csv,
            file_name="segmented_patients_with_advice.csv",
            mime="text/csv"
        )


# Main App Execution
if app_mode == "Heart Disease Prediction":
    heart_disease_prediction()
elif app_mode == "Patient Segmentation":
    patient_segmentation()
