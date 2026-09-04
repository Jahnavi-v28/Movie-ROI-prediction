# Movie ROI Prediction Using Explainable Machine Learning and Decision Support System

A machine learning project for predicting the **Return on Investment (ROI) of a movie before its release** using pre-release movie information.

The project combines **machine learning, explainable AI (SHAP), and a Decision Support System (DSS)** to not only estimate movie ROI, but also understand the factors behind the prediction and explore different production scenarios.

> **Published Research Paper:** International Journal of Creative Research Thoughts (IJCRT), Volume 14, Issue 8, August 2026  
> **Paper ID:** IJCRT2608242

---

## Overview

Movie production involves a large financial investment, and there is no guarantee that a movie will be profitable.

The goal of this project was to build a system that uses information available **before a movie is released** to estimate its expected ROI.

Instead of only giving a prediction, the project also looks at:

- Which factors are important for predicting ROI
- Why the model gives a particular prediction
- How changing production-related variables can affect the predicted ROI
- Whether a movie falls into a high, medium, or low investment-risk category

The complete source code is **not publicly available** in this repository.

---

## Objectives

The main objectives of the project were:

1. Predict movie ROI using pre-release information.
2. Compare different machine learning regression models.
3. Select the model with the best overall performance.
4. Use SHAP to understand the model's predictions.
5. Develop a Decision Support System for what-if analysis.
6. Provide an investment recommendation based on the predicted ROI.

---

## Project Workflow

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
ROI Calculation
      ↓
Feature Engineering
      ↓
Feature Encoding
      ↓
Train Multiple Regression Models
      ↓
Compare Model Performance
      ↓
Select Random Forest
      ↓
SHAP Explainability
      ↓
Decision Support System
      ↓
What-if Analysis
      ↓
Investment Recommendation
````


---

## Data and Preprocessing

The project used a movie dataset containing movie details, financial information, cast and crew information, keywords, and movie descriptions.

The different movie-related information was combined using a common movie identifier.

The target variable was **ROI**, calculated using:

```text
Profit = Revenue - Budget

ROI = (Revenue - Budget) / Budget
```

Records with missing or invalid budget and revenue values were removed because ROI could not be calculated for them.

Extreme outliers were also removed before model training.

---

## Feature Engineering

Several features were created from the available movie information.

### Numerical Features

* Budget
* Runtime
* Release year
* Release month
* Release quarter

### Categorical Features

* Genre
* Director
* Production company
* Language
* Country
* Franchise status
* Release season

### Text Features

The movie overview was also used as a text feature.

Categorical and text features were converted into numerical representations before training the models. The movie overview was transformed using **TF-IDF**.

---

## Machine Learning Models

Seven regression models were compared:

* Ridge Regression
* Decision Tree
* Random Forest
* Tuned Random Forest
* XGBoost
* LightGBM
* CatBoost

The dataset was divided into:

* **80% training data**
* **20% testing data**

The models were evaluated using:

* MAE
* RMSE
* R²
* Training Time

---

## Model Results

Random Forest achieved the best overall performance among the models tested.

| Model               |        MAE |       RMSE |         R² | Training Time (s) |
| ------------------- | ---------: | ---------: | ---------: | ----------------: |
| **Random Forest**   | **3.0862** | **6.1811** | **0.1872** |             14.83 |
| LightGBM            |     3.2550 |     6.2689 |     0.1639 |              1.82 |
| XGBoost             |     3.2252 |     6.2795 |     0.1611 |              7.05 |
| CatBoost            |     3.2222 |     6.3498 |     0.1422 |             17.73 |
| Tuned Random Forest |     3.2826 |     6.4708 |     0.1092 |             46.25 |
| Decision Tree       |     3.3176 |     6.6669 |     0.0544 |              0.65 |
| Ridge Regression    |     4.7927 |     7.2905 |    -0.1308 |              0.77 |

Random Forest had the lowest MAE and RMSE and the highest R² among the models tested, so it was selected for the explainability analysis and Decision Support System.

---

## Explainable AI with SHAP

A prediction alone does not explain why the model reached that result.

To understand the model, **SHAP (SHapley Additive exPlanations)** was used.

Two types of explanations were considered:

### Global Explanation

The global SHAP analysis looks at the dataset as a whole and shows which features have the greatest influence on the model's predictions.

### Local Explanation

The local explanation looks at an individual movie and shows how different features contribute to its predicted ROI.

---

## Important Features

The Random Forest feature importance analysis showed that the most important features included:

1. Budget
2. Release year
3. Franchise status
4. Runtime
5. Production company

Genre, director, and overview-based text features had comparatively smaller individual contributions.

The SHAP analysis also showed that franchise status, budget, release year, and runtime were important factors in the model's predictions.

---

## Decision Support System

The project also includes a **Decision Support System (DSS)** designed for what-if analysis.

Instead of only giving one prediction, the system allows a producer to select a movie and change certain production variables.

The variables that can be modified include:

* Budget
* Runtime
* Franchise status
* Release year
* Release month

The modified information is passed through the trained model to generate a new predicted ROI.

This allows different production scenarios to be compared before making an investment decision.

---

## Investment Recommendation

The Decision Support System categorizes the predicted ROI into three levels:

| Predicted ROI | Recommendation         | Risk        |
| ------------- | ---------------------- | ----------- |
| ROI < 0       | Do Not Invest          | High Risk   |
| 0 ≤ ROI < 1   | Invest with Caution    | Medium Risk |
| ROI ≥ 1       | Recommended Investment | Low Risk    |

These recommendations are intended as decision-support outputs and should not be treated as a guarantee of a movie's financial success.

---

## Key Findings

Some of the main findings from the project were:

* Random Forest performed better than the other regression models tested.
* Budget was the most important feature according to the Random Forest feature importance analysis.
* Franchise status, release year, runtime, and production company were also important.
* SHAP made it possible to understand both overall feature importance and individual predictions.
* The Decision Support System allowed production variables to be changed and different scenarios to be explored.
* Predicting movie ROI using only pre-release information is difficult because several important factors become known only after release.

---

## Limitations

Movie ROI depends on many factors that are difficult to capture before release.

The current project does not include factors such as:

* Marketing performance
* Critic reviews
* Audience reception
* Competing movie releases
* Social media trends
* Other information that becomes available after release

Because of this, the model should be viewed as a **decision-support tool rather than a system that guarantees financial success**.

---

## Future Work

The project can be extended by:

* Using a larger movie dataset
* Adding actor popularity
* Including marketing budget
* Adding audience reviews
* Using social media trends
* Improving the ROI prediction model
* Developing a web-based version of the Decision Support System

---

## Code Availability

The implementation used for this research is **not publicly available in this repository**

---

## Authors

### Vedula Jahnavi

School of Computing
SASTRA Deemed University, Thanjavur, India

### Anurag Shivam

School of Computing
SASTRA Deemed University, Thanjavur, India



The paper confirms the publication details, authors, methodology, model comparison, SHAP analysis, and DSS described above.   

**I would use this version rather than making the README overly fancy.** It reads like a student/researcher explaining what they actually did, and importantly, it doesn't pretend that the model is more accurate than the reported results show.
