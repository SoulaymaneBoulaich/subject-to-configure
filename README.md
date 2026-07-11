# LR Academic Project

A comprehensive machine learning project focused on linear regression analysis and predictive modeling of wellness and productivity metrics.

## Overview

This project implements multiple linear regression models to predict and analyze relationships between various factors and two primary target variables: Mental Wellness Index and Productivity Score. The analysis includes data preprocessing, exploratory data analysis, model training, validation, and comprehensive visualization of results.

## Project Structure

```
LR-Academic-Project/
├── data/
│   ├── input/
│   │   └── master_data_imputed.csv
│   ├── encoding/
│   │   ├── encoding_code.py
│   │   └── encoding.ipynb
│   └── vis/
│       ├── code_vis/
│       │   ├── code.ipynb
│       │   └── code.py
│       └── graphes_images/
├── ml/
│   ├── model/
│   │   ├── model_code.py
│   │   └── model.ipynb
│   └── pre_model/
│       ├── code_complet.py
│       ├── cross_validation.ipynb
│       └── metrics.ipynb
└── README.md
```

## Key Components

### Data Processing
- Data encoding and transformation in the `data/encoding/` directory
- Imputed master dataset handling and preprocessing
- Comprehensive feature engineering and preparation

### Visualization
- Exploratory data analysis visualizations including:
  - Distribution plots (donut charts, box plots, violin plots)
  - Correlation matrices and scatter plots
  - Age kernel density estimation plots
  - Stress level bar charts

### Machine Learning Models
- Linear regression implementation for wellness and productivity prediction
- Pre-processing pipeline for model preparation
- Cross-validation framework for model evaluation
- Performance metrics and model assessment

## Visualizations Generated

The project produces several analytical visualizations:
- graph1_donut.png - Distribution visualization
- graph2_boxplot.png - Statistical distribution analysis
- graph3_violin.png - Detailed distribution comparison
- graph4_correlation.png - Feature correlation matrix
- graph5_stress_bar.png - Stress level analysis
- graph6_scatter.png - Relationship scatter plots
- graph7_age_kde.png - Age distribution analysis
- reel_vs_predit_mental_wellness_index.png - Model predictions vs actual values
- reel_vs_predit_productivity_score.png - Productivity predictions vs actual values
- residus_mental_wellness_index.png - Residual analysis
- residus_productivity_score.png - Productivity residual analysis
- coefficients_mental_wellness_index.png - Feature coefficient visualization
- coefficients_productivity_score.png - Productivity feature coefficients

## Technology Stack

- Python 58.2%
- Jupyter Notebook 41.8%

## Main Libraries

- pandas - Data manipulation and analysis
- scikit-learn - Machine learning modeling
- matplotlib/seaborn - Data visualization
- numpy - Numerical computations

## Usage

1. Start with data preprocessing in `data/encoding/`
2. Explore data through visualizations in `data/vis/`
3. Run pre-processing pipeline in `ml/pre_model/`
4. Train and evaluate models in `ml/model/`
5. Review cross-validation results and metrics

## Features Analyzed

The models analyze relationships with Mental Wellness Index and Productivity Score, examining various demographic and behavioral factors through rigorous statistical and machine learning approaches.

## Model Evaluation

Cross-validation and comprehensive metrics evaluation are included to assess model performance, with residual analysis and prediction accuracy visualizations provided for both target variables.

## Repository Information

- Created: April 20, 2026
- Default Branch: main
- Status: Active
- License: Not specified

## Notes

This is an academic research project demonstrating applied machine learning and statistical analysis techniques for wellness and productivity prediction.
