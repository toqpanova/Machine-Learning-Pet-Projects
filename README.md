# Machine Learning Pet Projects

This repository contains three Jupyter notebooks showcasing core machine learning skills across regression and classification tasks. Each notebook highlights data preparation techniques, feature engineering, model tuning, and evaluation metrics.

---

## 📁 Notebook Summaries

### 01_insurance_regression.ipynb
- **Data Splitting:** Train / validation / test partitioning with controlled randomness.  
- **Scaling & Preprocessing:** Normalized numerical features to ensure fair model input.  
- **Regression Modeling:**  
  - Linear Regression: Fitted to predict healthcare expenses and computed RMSE.  
  - K-Nearest Neighbors Regression: Tested multiple k values, compared RMSE to select the best model.  
- **Feature Subset Analysis:** Iteratively removed predictors to measure their individual impact on performance.  
- **Final Evaluation:** Reported test-set RMSE for the chosen model.

### 02_attrition_classification.ipynb
- **Exploratory Analysis & Outlier Handling:**  
  - Visualized feature distributions by attrition status; clipped extreme values to reduce skew.  
- **Missing-Value Imputation:**  
  - Predicted missing “YearsInCurrentRole” using a KNN-based approach.  
  - Filled other gaps with sensible defaults based on domain insights.  
- **Categorical Encoding & Feature Creation:**  
  - Mapped travel frequency to numerical codes.  
  - Applied one-hot encoding for department categories.  
  - Created interaction features (e.g., JobSatisfaction × RelationshipSatisfaction, YearsInCurrentRole ÷ YearsAtCompany).  
- **Classification Modeling:**  
  - Stratified train/test split to maintain class balance.  
  - Trained a Random Forest Classifier; measured accuracy on held-out data.

### 03_heart_failure_classification.ipynb
- **Train/Test Split & Stratification:** Ensured consistent outcome proportions for “DEATH_EVENT.”  
- **Baseline & Cross-Validation:**  
  - Evaluated a Logistic Regression baseline via 5-fold F1 scoring.  
- **Tree-Based Model Tuning:**  
  - Iterated `max_depth` for Decision Tree and Random Forest classifiers; used cross-validation F1 scores to compare performance.  
- **Feature Engineering:**  
  - Built composite features (ejection_fraction × serum_creatinine, ejection_fraction ÷ serum_creatinine, serum_creatinine ÷ time, platelet ÷ serum_creatinine) to capture nonlinear relationships.  
- **Model Comparison:** Selected the optimal tree depth and method based on mean F1 results.

### 04_unsupervised_learning_kmeans_clustering.ipynb
Unsupervised clustering on product consumption data:
- Elbow Method to identify the optimal number of clusters (k=4).
- KMeans Clustering: Fitted model, predicted clusters, and appended labels to original dataset.
- Data Transformation & Aggregation: Used melt() and groupby() for reshaping and summarizing cluster-specific consumption patterns.
- Cluster Analysis: Identified most and least popular products within each cluster.
- Visualization: Plotted inertia vs. cluster count for optimal k selection using Matplotlib.
---

## 🔧 Skills Demonstrated

- **Data Preparation & Splitting**: Careful partitioning into training, validation, and test, with stratification to preserve class distribution.
- **Feature Scaling & Imputation**: Standardization of numerical predictors; KNN-driven imputation for missing values; outlier clipping.
- **Feature Engineering**: Creation of interaction and ratio-based features to uncover hidden patterns; one-hot encoding and mapping for categorical variables.
- **Model Training & Tuning**:  
  - **Regression**: Fitting linear and KNN regressors; selecting optimal neighbors (k) via RMSE comparison.  
  - **Classification**: Implementing Logistic Regression, Decision Trees, and Random Forests; hyperparameter tuning (`max_depth`) via cross-validated F1 scores.
- **Evaluation Metrics**: Root Mean Squared Error (RMSE), Accuracy, F1 Score; use of cross-validation to ensure robust performance estimates.
- **Visualization & EDA**: Generating distribution plots, histograms, and KDEs to understand data characteristics and identify outliers.
- **Reproducible Workflows**: Maintaining consistent random states for splits and model training; exporting final results for downstream use.
  **Unsupervised Learning & Clustering**: Applied KMeans clustering and selected optimal number of clusters using the Elbow Method.
   Used .predict() for label assignment and analyzed cluster-specific patterns through aggregation and reshaping techniques.
   Interpreted cluster structure via most/least common product features and visualized optimization with inertia plots.
---

Feel free to explore each notebook to see detailed code, comments, and insights into a full machine learning pipeline—from raw data to final evaluation.

