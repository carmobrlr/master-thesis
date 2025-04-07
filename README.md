# Master Thesis

## Predicting cardiomyocytes content using data-driven approaches

The repository contains all the code developed in this thesis and results obtained.

Cardiovascular diseases remain a leading cause of death worldwide, with cardiomyocyte (CM) loss as a major contributing factor. CMs are a promising tool for drug validation, autologous cell therapy, and cardiac disease modeling in vitro. They can be derived from human induced pluripotent stem cells through differentiation in stirred tank bioreactors, but this process remains highly variable.

Machine learning (ML) and biclustering algorithms can help optimize this process by identifying key bioprocess parameters influencing differentiation outcomes. This thesis applies biclustering to uncover significant data patterns and determine the most influential factors. Additionally, it compares supervised ML models to predict CM content using bioprocess data.

Biclustering analysis identifies cell density, dissolved oxygen concentration, and average pH gradient as key variables, with cell density increasing and oxygen levels decreasing from day 1. Biclustering reduces the original feature set by 75\%, selecting 25 features.

Five ML models—random forest, XGBoost, decision tree, support vector machine, and Gaussian Naive Bayes—are evaluated using feature sets obtained from biclustering and correlation coefficients. The best predictive performances are achieved by the decision tree with the ANOVA-derived feature set, the random forest with the biclustering feature set, and XGBoost with the ANOVA set. The decision tree with the ANOVA feature set achieves 93.3\% accuracy and 91.1\% precision.

This repository contains the initial data explorations, all the patterns obtained using the CCC-Biclustering algorithm and their evaluation using DISA tool, and the application of several ML models to the data. The models are optimized using a process of hyperparameter optimization and, in particular cases, resampling methods.
