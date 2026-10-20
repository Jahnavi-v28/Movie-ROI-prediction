# Movie ROI Prediction Using Explainable Machine Learning and Decision Support System

A machine learning research project that predicts movie Return on Investment (ROI) using pre-release movie metadata. The project combines regression models, SHAP-based explainability, and a web-based Decision Support System to explore movie investment scenarios.

**Published Research Paper:** International Journal of Creative Research Thoughts (IJCRT), Volume 14, Issue 8, August 2026  
**Paper ID:** IJCRT2608242

---

## Table of Contents

- [Overview](#overview)
- [Research Paper](#research-paper)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Dataset and Feature Engineering](#dataset-and-feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Explainable AI Using SHAP](#explainable-ai-using-shap)
- [Web Application](#web-application)
- [ROI Interpretation](#roi-interpretation)
- [Limitations](#limitations)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [Future Improvements](#future-improvements)
- [Authors](#authors)
- [Citation](#citation)

## Overview

Movie production requires substantial investment, but a movie's financial performance is uncertain. This project investigates how machine learning can estimate movie ROI using information available before release.

The research compares multiple regression models and selects Random Forest based on its performance on the evaluated dataset. SHAP is used to investigate the factors influencing predictions, while a Decision Support System supports what-if analysis of selected production variables.

A Flask web application has also been developed to accept movie details and display an ROI prediction.

**Note:** The prediction is an estimate based on historical data. It does not guarantee actual movie performance or financial returns.

## Research Paper

The published research paper is available in this repository:

**[Read the published paper — IJCRT2608242.pdf](./IJCRT2608242.pdf)**

The paper describes the dataset preparation, feature engineering, regression model comparison, SHAP analysis, and Decision Support System.

## Objectives

- Predict movie ROI using pre-release information.
- Compare different regression algorithms.
- Evaluate models using MAE, RMSE, R², and training time.
- Investigate model behaviour using SHAP explainability.
- Explore how changes to production-related variables affect predictions.
- Develop a web interface for generating movie ROI estimates.

## Methodology

The project follows this workflow:

1. Combine movie metadata, financial information, crew details, and keywords.
2. Clean the dataset and handle invalid financial records.
3. Calculate ROI as the regression target.
4. Engineer numerical, categorical, multi-label, and text features.
5. Encode the features for machine learning.
6. Split the dataset into training and testing sets.
7. Train and evaluate multiple regression models.
8. Select the best-performing model from the evaluated candidates.
9. Analyse model behaviour using SHAP.
10. Develop a Decision Support System for prediction and scenario exploration.

## Dataset and Feature Engineering

The research uses movie-related data containing financial information, movie metadata, production details, keywords, and overviews.

### Target Variable

The target is calculated as:

`ROI = (Revenue - Budget) / Budget`

Movies with invalid or non-positive budgets, or invalid revenue values, cannot provide a valid training target and are excluded during preprocessing. Extreme ROI outliers are also filtered as described in the research methodology.

### Features

**Numerical features**
- Production budget
- Runtime
- Release year
- Release month
- Release quarter, where used in the research pipeline

**Categorical and multi-label features**
- Genre
- Director
- Production company
- Original language
- Production country
- Spoken language
- Franchise indicator
- Release season

**Text features**
- Movie overview, transformed using TF-IDF with English stop words removed.

## Machine Learning Models

Seven regression approaches were evaluated:

- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

The dataset was divided into 80% training data and 20% testing data.

## Model Evaluation

The published research reported the following results:

| Model | MAE | RMSE | R² | Training Time (s) |
|---|---:|---:|---:|---:|
| **Random Forest** | **3.0862** | **6.1811** | **0.1872** | 14.83 |
| LightGBM | 3.2550 | 6.2689 | 0.1639 | 1.82 |
| XGBoost | 3.2252 | 6.2795 | 0.1611 | 7.05 |
| CatBoost | 3.2222 | 6.3498 | 0.1422 | 17.73 |
| Tuned Random Forest | 3.2826 | 6.4708 | 0.1092 | 46.25 |
| Decision Tree | 3.3176 | 6.6669 | 0.0544 | 0.65 |
| Ridge Regression | 4.7927 | 7.2905 | -0.1308 | 0.77 |

Random Forest achieved the lowest MAE and RMSE and the highest R² among the evaluated models, so it was selected for the final prediction system.

The R² value of 0.1872 also highlights the difficulty of predicting movie ROI from the available features. The model should therefore be treated as an experimental decision-support tool rather than a highly accurate financial forecasting system.

## Explainable AI Using SHAP

SHAP (SHapley Additive exPlanations) was used to interpret the trained model.

**Global explanations** examine feature contributions across the dataset, helping identify which features most influence model predictions overall.

**Local explanations** examine individual predictions to understand how the input features contribute to a particular model output.

The research discusses budget, franchise status, release year, runtime, and production company among the important features identified through feature importance and SHAP analysis.

## Web Application

The project includes a Flask-based web application connected to a saved Random Forest model.

The current application supports:

- Entering movie budget and runtime.
- Selecting franchise status and release information.
- Entering language, genre, director, company, country, and keyword details.
- Providing a movie overview.
- Generating a predicted ROI.
- Displaying the prediction as a percentage alongside the raw model output.
- Showing a disclaimer that predictions are estimates and actual results may vary.

The web application is a practical interface to the trained model. It does not guarantee profitability or replace independent financial analysis.

## ROI Interpretation

Because the target is defined as:

`ROI = (Revenue - Budget) / Budget`

the model output is a ratio.

For example, a predicted ROI of `1.5` corresponds mathematically to 150% ROI. If a movie had a budget of ₹10 crore, that ratio would imply ₹15 crore in profit and ₹25 crore in revenue if the estimate were realised.

This is an illustration of the formula, not a forecast of actual revenue.

## Limitations

Movie performance depends on many factors that may be unavailable before release, including marketing effectiveness, audience reception, reviews, competing releases, and changing audience preferences.

Additional limitations include:

- The model's reported R² is relatively low.
- Predictions are influenced by the quality and coverage of the training data.
- Unseen categories or missing metadata may reduce prediction reliability.
- Historical box-office ROI does not necessarily represent OTT revenue or total investment returns.
- A model prediction alone is insufficient for an investment decision.

## Technology Stack

The research and application use the following technologies where applicable:

- **Python** — data processing and model development
- **Pandas and NumPy** — data manipulation
- **Scikit-learn** — preprocessing and regression models
- **TF-IDF** — text feature extraction
- **SHAP** — model explainability
- **Flask** — web application
- **HTML and CSS** — user interface
- **Joblib** — saving and loading trained model artifacts

## Project Structure

The repository will contain the published paper, research materials, and web application files as they are added.

```text
Movie-ROI-prediction/
├── IJCRT2608242.pdf
├── README.md
├── app.py
├── model_utils.py
├── requirements.txt
├── .gitignore
├── models/
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── feature_columns.pkl
│   └── encoder files
├── templates/
│   └── index.html
└── notebooks/
    └── research and model development notebooks
```

This is the intended structure; the repository should only be considered complete once the corresponding files have actually been uploaded.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/Jahnavi-v28/Movie-ROI-prediction.git
cd Movie-ROI-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

These instructions assume that the required model artifacts and dependency list are present in the repository.

## Future Improvements

- Improve generalisation and prediction performance.
- Evaluate additional features and more representative datasets.
- Expand the web interface for scenario comparisons.
- Integrate SHAP explanations directly into the web application.
- Add automated tests for preprocessing and predictions.
- Improve handling of missing and previously unseen categories.
- Explore uncertainty estimates to communicate prediction reliability.

## Authors

**Vedula Jahnavi**  
School of Computing, SASTRA Deemed University, Thanjavur, India

**Anurag Shivam**  
School of Computing, SASTRA Deemed University, Thanjavur, India

## Citation

If you reference this research, please cite the published paper using its official bibliographic details.

**Paper:** IJCRT2608242  
**Journal:** International Journal of Creative Research Thoughts (IJCRT)  
**Volume:** 14, Issue 8  
**Publication:** August 2026

Please consult the attached paper PDF for the full citation and publication details.
