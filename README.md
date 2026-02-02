# Master Thesis  
## Predicting Cardiomyocyte Content Using Data-Driven Approaches

This repository contains the code and results developed for my Master's thesis, focused on predicting cardiomyocyte (CM) content during stem cell differentiation using data-driven methods.

---

## Project Motivation
Cardiovascular diseases are a leading cause of death worldwide, with cardiomyocyte loss playing a major role. Cardiomyocytes derived from human induced pluripotent stem cells are a promising solution for drug testing and disease modelling. However, the differentiation process in stirred-tank bioreactors is highly variable.

This project explores how **biclustering and machine learning** can be used to identify key bioprocess parameters and improve the prediction of cardiomyocyte content.

---

## Methodology
The project follows a structured data science pipeline:

1. Exploratory data analysis and correlation assessment  
2. Pattern discovery using **CCC-Biclustering (BiGGEstTS)**  
3. Pattern evaluation using the **DISA** tool  
4. Feature selection based on significant biclusters  
5. Supervised machine learning modelling and evaluation  

---

## Key Results
- Biclustering identified **cell density, dissolved oxygen concentration and pH gradient** as key variables influencing differentiation.
- Feature space was reduced by **~75%**, selecting 25 relevant features.
- Five ML models were evaluated:  
  *Decision Tree, Random Forest, XGBoost, SVM and Gaussian Naive Bayes.*
- Best performance:
  - **Decision Tree + ANOVA feature set**:  
    - Accuracy: **93.3%**  
    - Precision: **91.1%**
  - Random Forest and XGBoost also achieved strong results with selected feature sets.

---

## Repository Structure
```text
code/
├── 01_exploration            # Initial data analysis and visualizations
├── 02_biclustering           # Pattern discovery with BiGGEstTS
├── 03_pattern_evaluation     # DISA-based pattern evaluation
├── 04_feature_engineering    # Feature selection and comparison
├── 05_modeling               # Machine learning models and evaluation
└── datasets                  # Raw and processed datasets

