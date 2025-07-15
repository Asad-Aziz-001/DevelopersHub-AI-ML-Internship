# DevelopersHub-AI-ML-Internship

✅ Internship Project Summary

📘 Tasks: Dataset Exploration, Stock Price Prediction, and Heart Disease Classification

# *🔷 Task 1: Exploring and Visualizing the Iris Dataset*

🎯 Objective:
Understand the structure, patterns, and distribution of data using visualization.

📦 Dataset:
Iris Dataset (150 records, 3 classes of flowers)

🔁 Steps & Working:
Load Dataset using pandas and seaborn:

python
Copy
Edit
import seaborn as sns
df = sns.load_dataset('iris')
Basic Info & Inspection:

.shape, .columns, .head() to inspect the structure

.info() and .describe() for types and stats

Visualizations:

Scatter plot of sepal/petal relationships (seaborn.scatterplot)

Histogram to observe distribution (seaborn.histplot)

Box plot to detect outliers in numeric features

✅ Outcome:

Learned to visually analyze data, detect outliers, and observe trends between flower species based on dimensions.

# *🔷 Task 2: Predicting Stock Prices (Short-Term Forecasting)*

🎯 Objective:
Predict the next day’s closing price for Tesla stock using historical data.

📦 Dataset:
Tesla stock from Yahoo Finance (via yfinance or Kaggle mirror)

🔁 Steps & Working:
Data Loading:

Fetched historical Tesla stock prices (Open, High, Low, Close, Volume)

Feature Engineering:

Target = next day’s Close using .shift(-1)

Dropped NaNs and selected features: ['Open', 'High', 'Low', 'Volume']

Modeling:

Trained Linear Regression and Random Forest Regressor

Used train_test_split (no shuffling due to time series)

Evaluation:

Compared models using RMSE and R² Score

Visualized actual vs predicted close price

Linear Regression performed better with higher R² (~0.97)

✅ Outcome: 

Successfully predicted short-term prices using regression. Learned time-series handling and comparison of linear vs ensemble models.

# *🔷 Task 3: Heart Disease Prediction (Binary Classification)*

🎯 Objective:
Predict whether a patient is at risk of heart disease based on medical data.

📦 Dataset:
UCI Heart Disease Dataset (Kaggle mirror)

🔁 Steps & Working:
Data Cleaning:

Checked missing values

Encoded categorical features using pd.get_dummies()

Exploratory Data Analysis (EDA):

Target distribution via countplot

Feature distributions via histograms and boxplots

Correlation heatmap (only numeric columns)

Modeling:

Trained Logistic Regression and Decision Tree Classifier

Used one-hot encoded data

Evaluation:

Used Accuracy, ROC-AUC, Confusion Matrix, and Classification Report

Visualized ROC curves for both models

Logistic Regression showed slightly better generalization

Feature Importance:

Decision Tree: .feature_importances_

Logistic Regression: .coef_

✅ Outcome:

Built interpretable ML models for medical classification. Identified key risk indicators such as chest pain type, ST depression, and max heart rate.

#*🏠 Notebook: Task 6: House Price Prediction*

✅ Core Objective

Predict house prices using key features: size (area), bedrooms, location, and target variable price.

🔁 Workflow and Key Steps

Data Loading & Initial Inspection
Loaded the Kaggle dataset and explored columns and data statistics to confirm availability of critical variables.

Data Preprocessing

Performed one-hot encoding on Location, Condition, and Garage to convert categorical features to numeric.

Confirmed no missing values in main features.

Feature & Target Definition

Features (X): area, bedrooms, and dummy variables for Location, Condition, Garage.

Target (y): continuous price.

Train-Test Split

Applied an 80/20 split for training and test datasets to evaluate model performance effectively.

Feature Scaling

AWS: Standardized numeric features using StandardScaler before applying Linear Regression.

Model Training

Lasso Regression was trained to perform regularized linear fitting.

Gradient Boosting Regressor (GBR) was trained as a nonlinear ensemble model.

Model Evaluation

For each model, calculated:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Visualized results with plots showing predicted vs actual prices for the top 50 samples.

📊 Performance Summary
Lasso Regression:

MAE ≈ 241 K

RMSE ≈ 278 K

Gradient Boosting:

MAE ≈ 243 K

RMSE ≈ 284 K

🎯 Lasso slightly outperformed, based on those metrics.

✅ Conclusion
You’ve successfully built and evaluated two robust regression pipelines focusing on core property features:

Perfectly aligned with the task requirements

Scored strong baseline performance with interpretable metrics
