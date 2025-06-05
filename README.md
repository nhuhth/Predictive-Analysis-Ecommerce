# Predictive Analysis in E-commerce
## Team Members
- Abdurehman Asfaw (448665)  
- Ayyub Orujzade (466757)  
- Bhupender Bhupender (466758)  
- Ho Thi Hoang Nhu (466503)

This project focuses on predicting whether a customer would recommend a product based on their review. We started by reproducing the project by [janhavi-giri](https://github.com/janhavi-giri/Predictive-Analysis-Ecommerce), then made enhancements to improve model performance and ensure the whole process is fully reproducible.

---

## Dataset

- **Source**: [Kaggle - Women's E-Commerce Clothing Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)  
- **Rows**: 23,486  
- **Target Column**: `Recommended IND` (1 = Recommended, 0 = Not Recommended)

---

## Objectives

- Develop a binary classification model to predict product recommendations.
- Enhance reproducibility.
- Improve upon the original project using better:
  - Data preprocessing  
  - Feature engineering  
  - Feature selection  
  - Model optimization

---

## Workflow Overview

### 1. **Data Preprocessing & Feature Engineering** (`DP_FE.ipynb`)
- Dropped rows with missing or irrelevant values.
- Cleaned and lemmatized text.
- Created new features:
  - `review_length` (from cleaned review text)
  - `sentiment_score` (using VADER)
- Used one-hot encoding and TF-IDF.
- Saved processed train/test datasets.

### 2. **Feature Selection & Modeling** (`FS_Model.ipynb`)
- Scaled numeric features.
- Removed highly correlated variables (> 0.85).
- Selected top 300 features using Random Forest importance + Mutual Information.
- Tuned Random Forest model using GridSearchCV.

---

## 📊 Model Performance

On the test set:

| Metric       | Score   |
|--------------|---------|
| Accuracy     | 0.95    |
| ROC AUC      | 0.9899  |
| F1-score (0) | 0.87    |
| F1-score (1) | 0.97    |

**Confusion Matrix**:
- True Positives: 3052  
- False Positives: 53  
- False Negatives: 131  
- True Negatives: 638  

---

## 🆚 Comparison with Original

| Category           | Original Project        | Our Version                      |
|-------------------|-------------------------|----------------------------------|
| Text Processing    | Basic                   | Cleaned, lemmatized, sentiment   |
| Feature Engineering| Limited                 | New features + TF-IDF            |
| Feature Selection  | Not systematic          | Combined RF + MI                 |
| Model Tuning       | Not specified           | GridSearchCV                     |

---

## 🔁 Reproducibility

### 1. Clone the repo
```bash
git clone https://github.com/abdiyimer/Predictive-Analysis-Ecommerce
cd Predictive-Analysis-Ecommerce
```
### 2. Set up environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt


