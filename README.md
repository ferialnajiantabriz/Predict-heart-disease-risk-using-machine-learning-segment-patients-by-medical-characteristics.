# Heart Disease Prediction Using Data Mining Algorithms and Patient Segmentation

## 📌 Overview
This project presents a comprehensive heart disease risk prediction and patient segmentation system using machine learning and clustering techniques. It is designed to assist healthcare providers in diagnosing heart disease and understanding patient patterns through a user-friendly interface and insightful visualizations.

## 🎯 Objectives
- Predict heart disease risk using machine learning models: Decision Tree, Logistic Regression, and k-Nearest Neighbors (k-NN).
- Segment patients into distinct risk groups using K-Means clustering.
- Provide an intuitive GUI for data input and visualization using Streamlit.

## 🧠 Machine Learning Models
All models were implemented from scratch to enhance interpretability:
- **Decision Tree:** High accuracy and transparency; chosen as the best-performing model.
- **Logistic Regression:** Captures linear relationships with strong AUC score.
- **k-NN:** Distance-based classifier, good for baseline comparison.
- **Naïve Bayes:** Included for comparative analysis.

Performance metrics include:
- Accuracy
- Precision, Recall, F1 Score
- G-Mean
- Confusion Matrix and ROC Curve

## 🧪 Dataset
- **Source:** [UCI Heart Disease Dataset (via Kaggle)](https://www.kaggle.com/johnsmith88/heart-disease-dataset)
- **Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, ECG, Heart Rate, Angina, Oldpeak, ST Slope, Major Vessels, Thalassemia

## 🔍 Data Preprocessing
- **Normalization** of continuous features
- **One-hot encoding** for categorical features
- **Outlier removal** using IQR method
- Final dataset size: 946 records after cleaning

## 📊 Exploratory Data Analysis
- Histograms and boxplots for feature distribution and outliers
- Count plots for categorical features
- Correlation heatmap to identify key variables
- Feature importance from multiple models

## 👥 Patient Segmentation
- Custom **K-Means** clustering (from scratch)
- Optimal number of clusters (K=4) chosen via Elbow Method
- Segments explained with recommendations for each group
- Cluster interpretation using:
  - Pair plots
  - PCA visualizations
  - Silhouette analysis
  - Distribution analysis of categorical features

## 🖥️ Streamlit Interface
- Manual and batch input support
- Real-time and CSV-based predictions
- Cluster assignment and recommendations
- Downloadable results
- Dynamic visualizations (confusion matrix, ROC, pair plots, etc.)

## 🚀 Deployment
- Currently deployed locally via Streamlit
- Ready for cloud deployment for real-world accessibility

## 📁 Project Structure
heart-disease-prediction/
├── data/ # Sample CSV files
├── models/ # Custom ML model implementations
├── preprocessing/ # Data cleaning and transformation
├── ui/ # Streamlit UI code
├── results/ # Visualizations and reports
├── main.py # Entry point for Streamlit app
└── README.md # This file



## 📚 References
- Almustafa (2020)
- Ali et al. (2021)
- Bhatla and Jyoti (2012)
- Chandrasekhar and Peddakrishna (2023)
- Dey and Rautaray (2014)
- Soni et al. (2011)

## 🤝 Contributors
- Ferial Najiantabriz – University of Oklahoma
- Subankar Chowdhury – University of Oklahoma
- Ujwala Vasireddy – University of Oklahoma

## 📬 Contact
Feel free to reach out: najiantabriz.ferial@gmail.com
