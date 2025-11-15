# Solar-Power-Generation-Predictor
Solar energy output prediction using optimized XGBoost regression with scaled environmental features and an interactive Streamlit UI.

## 🔗 Live Solar Power Prediction App
Click below to use the deployed application:

🌐 (https://solar-power-generation-predictor.streamlit.app/)


Machine Learning Regression Project using XGBoost & Streamlit

📌 Project Overview

This project predicts solar energy generation (in Joules per 3-hour period) using environmental parameters such as:

* Distance to solar noon

* Temperature

* Sky cover

* Visibility

* Humidity

* Wind speed

The goal is to build a highly accurate regression model, visualize insights through EDA, and provide a Streamlit-based web application for real-time predictions.

🚀 Key Features
- Exploratory Data Analysis (EDA)

- Feature distribution analysis

- Correlation heatmaps

- Outlier inspection

- Insights and interpretations for model building

🤖 Machine Learning Models

used multiple regression models and choosed the best performing model based on Rsquare

- Regressors tested:

- Linear Regression

- Ridge Regression

- Lasso Regression

- Random Forest

- LightGBM

- XGBoost (Best Performing Model: R² ≈ High Accuracy)

Additional optimization:

- Cross-validation

- Hyperparameter tuning

- Ensemble comparison

🌐 Interactive Web App

Built using Streamlit featuring:

Dark futuristic UI

Radar loading animation

Dropdowns & sliders

Line chart comparing predicted vs highest recorded power

Real-time energy prediction
