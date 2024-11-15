# Gold Price Prediction: Regression Models

This project aims to predict gold prices using various regression models. It explores multiple machine learning algorithms to find the best model for predicting gold prices based on historical data and additional financial indicators.

## Project Overview

The dataset used for this project contains historical gold prices along with other financial indicators like the S&P 500 Index (SPX), Oil Fund (USO), Silver Price (SLV), and the EUR/USD exchange rate. The goal is to build a predictive model that can forecast future gold prices based on these features.

## Table of Contents

1. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Modeling](#modeling)
4. [Evaluation](#evaluation)
5. [Conclusion](#conclusion)
6. [Requirements](#requirements)
7. [License](#license)

## Data Collection and Preprocessing

1. **Dataset Loading**: The dataset (`gold_price_data.csv`) is loaded from a Google Drive path.
2. **Data Cleaning**: The 'Date' column is converted into datetime objects and useful features such as Year, Month, and Day are extracted.
3. **Missing Values**: Missing data is checked for and handled appropriately.

## Exploratory Data Analysis (EDA)

The EDA section includes:
- Visualization of the distribution of gold prices using histograms and KDE plots.
- A time series plot to show how gold prices have evolved over the years.
- Correlation analysis to explore relationships between features.

### Key EDA Visualizations:
- **Distribution of Gold Prices**: A histogram with KDE showing the spread of gold prices.
- **Gold Prices Over Time**: A line plot showing how gold prices changed by year and month.
- **Histograms**: Individual histograms for gold prices, SPX, USO, SLV, and EUR/USD.
- **Correlation Matrix**: A heatmap showing the correlation between features.

## Modeling

Multiple regression models are trained and tuned to predict gold prices:

1. **Linear Regression**
2. **Ridge Regression** (with GridSearchCV for hyperparameter tuning)
3. **Support Vector Regression (SVR)** (with GridSearchCV for hyperparameter tuning)
4. **Decision Tree Regression** (with GridSearchCV for hyperparameter tuning)
5. **Random Forest Regression** (with GridSearchCV for hyperparameter tuning)

### Hyperparameter Tuning:
GridSearchCV is used to optimize hyperparameters for each model, ensuring the best performance.

## Evaluation

Each model is evaluated based on:
- **Mean Squared Error (MSE)**
- **R-squared (R²) Score**

Results from all models are printed, allowing comparison of performance.

## Conclusion

The project provides insights into the performance of various machine learning regression models for predicting gold prices. By comparing metrics like MSE and R², the most effective model can be selected for future predictions.

## Requirements

To run this project, you need the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`

These libraries can be installed using pip:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
