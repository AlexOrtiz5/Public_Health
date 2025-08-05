# Public_Health
  Project status(Active)

# Project objective
  This project undertakes a thorough analytical and predictive exploration of a public health dataset, aiming to accurately predict the presence of heart disease dataset sourced from Kaggle, specifically from the "Heart Disease Dataset" available at https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset. Our approach has evolved into a robust, multi-faceted methodology to ensure the reliability and fairness of our findings. We began with rigorous data preparation, which included a deep dive into outlier detection and handling using the Interquartile Range (IQR) method and addressing missing data through imputation. For advanced predictive modeling, we have employed model ensembling and stacking to combine the strengths of multiple models for improved accuracy and stability. A critical component of our work is the fairness and ethical analysis, where we proactively identify and visualize potential model bias related to sensitive attributes like sex and age. To ensure a reliable estimate of our model's performance, we implemented cross-validation to prevent overfitting. Furthermore, we specifically addressed data challenges like class imbalance using strategies like SMOTE and class weights. Moving forward, we will continue to perform feature importance analysis, hypothesis testing, and explore patient segments through unsupervised clustering to derive actionable insights from this critical health data.

# Methods
  List with methods:
  - Data Cleaning and Preprocessing:
    - Handling Missing Values (Imputation)
    - Removing Duplicate Rows
    - Correcting Data Types
  - Descriptive Statistics:
    - Calculating Summary Statistics (mean, median, std, min, max, quartiles) for numerical features.
    - Generating Frequency Counts and Percentages for categorical features.
  - Exploratory Data Analysis (EDA):
    - Visualizing Distributions (Histograms, KDE plots for numerical data; Count plots for categorical data).
    - Analyzing Relationships between Features and Target (Box plots, Violin plots, Count plots with hue, Stacked Bar Charts).
    - Creating Correlation Heatmaps.
    - Generating Pair Plots (for numerical features).
  - Predictive Modeling (Classification):
    - Data Splitting (Training and Testing sets).
    - Feature Scaling.
    - Training Classification Models (e.g., Logistic Regression, Random Forest Classifier).
    - Model Evaluation (Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix, Classification Report).
  - Feature Importance Analysis:
    - Extracting Model-based Feature Importances (from Random Forest).
    - Calculating Permutation Importance.
    - Visualizing Feature Importances.
  - Segmentation/Clustering (Unsupervised Learning):
    - Determining Optimal Number of Clusters (Elbow Method).
    - Applying K-Means Clustering.
    - Analyzing Cluster Characteristics (mean feature values per cluster, categorical distributions).
    - Visualizing Clusters (using PCA for dimensionality reduction).
  - Hypothesis Testing:
    - Independent Samples T-tests (comparing numerical means between two groups).
    - Chi-squared Tests of Independence (assessing association between two categorical variables).
  - Comprehensive Data Visualization:
    - Creating various plots to illustrate data distributions and relationships, often with the target variable as a focus.

# Technologies 
  List with used technologies:
  - Python: The core programming language for all data processing, analysis, and modeling.
  - Pandas: Utilized extensively for data loading, manipulation, cleaning, and structuring.
  - NumPy: Employed for fundamental numerical operations, particularly in conjunction with Pandas DataFrames.
  - Matplotlib: Used for creating static, interactive, and animated visualizations in Python.
  - Seaborn: Built on Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics.
  - Scikit-learn (sklearn): A comprehensive machine learning library used for:
    - Data splitting (train_test_split)
    - Feature scaling (StandardScaler)
    - Classification models (LogisticRegression, RandomForestClassifier)
    - Model evaluation metrics
    - Clustering (KMeans)
    - Dimensionality reduction (PCA)
    - Permutation Importance (permutation_importance)
  - SciPy: A scientific computing library, specifically used for statistical functions like t-tests and chi-squared tests (scipy.stats).

# Project Description
  The dataset central to this project is the "Heart Disease Dataset," openly available on Kaggle and accessible directly via the URL: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset. This public health-oriented collection initially features patient attributes such as age, sex, trestbps (resting blood pressure), chol (serum cholesterol), fbs (fasting blood sugar), restecg (resting electrocardiographic results), thalach (maximum heart rate achieved), exang (exercise-induced angina), oldpeak (ST depression induced by exercise relative to rest), slope (the slope of the peak exercise ST segment), ca (number of major vessels colored by fluoroscopy), and thal (thalassemia type). It also includes one-hot encoded representations of chest pain type (cp_0, cp_1, cp_2, cp_3), and crucially, a binary target variable indicating the presence (1) or absence (0) of heart disease.

  Beyond these original attributes, our project has significantly enriched the dataset through advanced feature engineering. This process introduced several new characteristics, including polynomial features (e.g., age^2, age trestbps, chol^2), manual interaction terms (such as age_x_chol and thalach_x_exang), binned age groups (age_group_X after one-hot encoding), and ratio features like chol_to_trestbps_ratio. We also combined the one-hot encoded chest pain types into a single cp_type categorical feature. These expanded characteristics make the dataset even more comprehensive for classification tasks aimed at predicting heart disease and for in-depth exploratory analysis to understand the underlying health factors.

# Steps
  - Data Cleaning and Preprocessing:
    - Step: Loaded the health.csv dataset, identified and handled missing values (e.g., ?, N/A) by imputing numerical columns with their median and categorical columns with their mode. Ensured all columns were converted to appropriate data types (e.g., Int64 for nullable integers, float for continuous numericals). Duplicate rows were also identified and removed.
    - Insight: The initial dataset required robust cleaning, particularly in handling non-standard missing value indicators and ensuring correct data types. This step was crucial to prevent errors in subsequent analyses and ensure data integrity. The presence of duplicates highlighted the need for careful data governance.
  - Descriptive Statistics:
    - Step: Calculated summary statistics (mean, median, standard deviation, min/max, quartiles) for numerical features and frequency counts/percentages for categorical features.
    - Insight: This provided a foundational understanding of the dataset's characteristics. For instance, we observed the average age of patients, the distribution of different chest pain types, and the overall prevalence of the target condition. This initial overview guided our focus for more in-depth analyses.
  - Exploratory Data Analysis (EDA):
    - Step: Generated various visualizations including histograms, KDE plots for numerical distributions, count plots for categorical distributions, box plots and violin plots to compare numerical features across target groups, and stacked bar charts to show categorical feature distributions by target. A correlation heatmap of numerical features was also created.
    - Insight: EDA revealed visual patterns and potential relationships. For example, we could visually infer that certain ranges of age, chol, or thalach might be more common in patients with heart disease. The correlation heatmap showed us which numerical features had stronger linear relationships with each other and with the target variable.
  - Predictive Modeling (Classification):
    - Step: Split the preprocessed data into training and testing sets, performed feature scaling on numerical features, and trained two classification models: Logistic Regression and Random Forest Classifier. The models were evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, ROC AUC, and Confusion Matrix.
    - Insight: Both models demonstrated a reasonable ability to predict heart disease. The Random Forest Classifier often showed slightly better performance, indicating its capability to capture more complex, non-linear relationships in the data. Evaluating precision and recall was particularly important for this health context, as the cost of false negatives (missing a disease case) can be higher than false positives.
  - Feature Importance Analysis:
    - Step: Utilized the trained Random Forest model's built-in feature importances and also calculated Permutation Importance on the test set. The results were visualized using bar plots.
    - Insight: This step was critical for interpretability. We identified key features that significantly influenced the model's predictions, such as cp (chest pain type), thalach (max heart rate), exang (exercise-induced angina), oldpeak, ca (number of major vessels), and thal (thalassemia). These insights suggest which clinical measurements are most indicative of heart disease within this dataset.
  - Segmentation/Clustering (Unsupervised Learning):
    - Step: Applied K-Means clustering to the features (excluding the target variable) after scaling. The Elbow Method was used to help determine a suitable number of clusters. The characteristics of the resulting clusters were then analyzed by examining the mean feature values within each cluster and visualizing them using PCA.
    - Insight: This analysis revealed natural groupings of patients based purely on their health attributes. We could characterize these clusters (e.g., one cluster might represent patients with generally healthier profiles, while another might represent those with more severe symptoms, regardless of their actual heart disease diagnosis). This provides a different lens for understanding patient heterogeneity.
  - Hypothesis Testing:
    - Step: Performed Independent Samples T-tests to compare the means of numerical features between the 'disease' and 'no disease' groups. Chi-squared tests of independence were conducted to assess associations between categorical features and the target variable.
    - Insight: Hypothesis testing provided statistical validation for many of the relationships observed during EDA. For example, a significant p-value from a t-test on chol would confirm that cholesterol levels are indeed statistically different between patients with and without heart disease. Similarly, a significant chi-squared test for sex and target would indicate a statistically significant association between gender and heart disease prevalence in this dataset.

# Conclusion
  In conclusion, this comprehensive project successfully navigated through various stages of data analysis, providing valuable insights into the factors influencing heart disease based on the Kaggle "Heart Disease Dataset." Through meticulous data cleaning and preprocessing, we established a reliable foundation for subsequent analyses. Our exploratory data analysis (EDA) revealed crucial distributions and initial relationships, visually confirming that several attributes, such as age, cholesterol, and maximum heart rate, are distinctly distributed between individuals with and without heart disease. The predictive modeling phase demonstrated the feasibility of accurately identifying individuals at risk, with the Random Forest Classifier proving to be a robust tool for this classification task. Furthermore, the feature importance analysis underscored the critical roles of chest pain type, maximum heart rate, exercise-induced angina, and the number of major vessels in predicting heart disease, offering actionable insights for clinical assessment. Unsupervised clustering unveiled natural patient segments, providing a novel perspective on patient heterogeneity beyond the binary disease outcome. Finally, hypothesis testing statistically validated many of these observed relationships, reinforcing the significance of key features. Overall, this project not only built effective predictive models but also deepened our understanding of the dataset's characteristics and the complex interplay of factors contributing to heart disease.
  
# Contact
  linkedin, github, medium, etc 