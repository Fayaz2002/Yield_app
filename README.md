# Yield_app

# 📌 Project Overview
This project presents a data-driven approach to forecast rice crop yield in India using machine learning models. The primary objective was to assess the influence of climatic and agronomic factors on rice productivity and generate accurate, interpretable predictions. The work formed the core of my MSc dissertation titled "Predictive Modeling of Rice Crop Yield in India: Analyzing the Impact of Climatic and Agronomic Factors."

## App Link
https://riceyield.streamlit.app/

The study utilized historical data from 1950 to 2023 and implemented various machine learning techniques to model the relationships between environmental/agricultural inputs and yield output. To improve usability and practical relevance, the predictive system was also deployed as a Streamlit web application, allowing real-time user interaction.

# 🎯 Objectives
Analyze how climatic and agronomic factors influence rice yield.

Build robust machine learning models to forecast yield.

Evaluate model performance using appropriate metrics.

Provide insights to support agricultural decision-making.

Deploy an interactive web app for real-time predictions and analysis.

# 📂 Dataset
Time Range: 1950–2023

Features Used:

TEMP: Average yearly temperature

RAINFALL: Total annual rainfall

SIZE: Area used for rice cultivation (in million hectares)

AUI: Area under irrigation (in million hectares)

NITROGEN, PHOSPHORUS, POTASSIUM: Fertilizer usage (kg/hectare)

RYIELD: Rice crop yield (target variable, kg/hectare)

# ⚙️ Methodology
1. Data Preprocessing
Handled missing values and outliers

Scaled and normalized features as needed

Ensured consistency in units and time alignment

2. Feature Engineering
Derived key agronomic indicators

Explored correlation and interaction effects between features

3. Model Selection
Compared models: Linear Regression, Random Forest, ARIMAX, ElasticNet, and Lasso

Applied time-series models where appropriate for yield forecasting

4. Hyperparameter Tuning
Performed grid search for models like Random Forest

Tuned parameters like n_estimators, max_depth, min_samples_split, etc.

5. Model Evaluation
Metrics used:

R² (Coefficient of Determination)

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Visualized model fit: Actual vs Predicted plots over time

# 🧠 Key Models Used
Random Forest Regressor: Performed best in terms of predictive accuracy but showed slight overfitting due to small dataset.

ARIMAX: Provided a well-generalized model leveraging past trends with external factors (exogenous variables).

ElasticNet: Balanced model that reduced overfitting while maintaining interpretability.

# 🌐 Streamlit Web App
An interactive web interface was created to:

Allow users to input yearly values for climate and fertilizer use

Generate real-time yield predictions

Provide impact analysis, showing which variables contributed most to the predicted change

Offer qualitative and quantitative recommendations, such as:

“Add 10 kg/hectare of nitrogen”

“Increase irrigated area by 1.5 million hectares”

# 🔍 Insights
Rainfall and temperature had significant but nonlinear impact on yield.

Nitrogen use showed strong positive correlation, but returns diminished after a point.

Irrigated area consistently boosted yield, especially in drier years.

# 🔮 Future Work
Mobile App Deployment: Enable offline access for field use.

Multi-Crop Expansion: Extend the model to support crops like wheat and maize.

State-wise Forecasting: Integrate regional models to account for local conditions.

Quantified Recommendations: Improve current suggestions with exact amounts (e.g., area, fertilizer).

# Images
![image](https://github.com/user-attachments/assets/83801bc2-5a00-470c-afd5-3ce3e174768d)

