# Reproducible Research project in E-commerce

# Predictive Analysis in E-commerce: A Reproducible Research and Enhancement of Product Recommendation

This data science project focuses on building a predictive model to determine whether a customer will recommend a product, based on review data from an online clothing retailer. This project is a **fork, reproduction, and enhancement** of janhavi-giri's [Predictive Analysis Ecommerce](https://github.com/janhavi-giri/Predictive-Analysis-Ecommerce) project. This version can be found at: [https://github.com/abdiyimer/Predictive-Analysis-Ecommerce](https://github.com/abdiyimer/Predictive-Analysis-Ecommerce). This project places special emphasis on **enhancing reproducibility, applying more detailed data processing and feature engineering techniques, and systematically optimizing the model.**

## Dataset

* **Source:** [Women's E-Commerce Clothing Reviews on Kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews)
* **Description:** The dataset contains 23,486 rows and 10 original feature variables, where each row corresponds to a customer review. The target variable is `Recommended IND` (whether the customer recommends the product).
* **Original Feature Variables:**
    1.  `Clothing ID`: Product ID.
    2.  `Age`: Reviewer's age.
    3.  `Title`: Title of the review.
    4.  `Review Text`: Body of the review.
    5.  `Rating`: Product rating (1-worst to 5-best).
    6.  `Recommended IND`: Recommendation indicator (1 = recommended, 0 = not recommended).
    7.  `Positive Feedback Count`: Number of positive feedback for the review.
    8.  `Division Name`: High-level division name of the product.
    9.  `Department Name`: Department name of the product.
    10. `Class Name`: Class name of the product.
    (Feature descriptions sourced from the original project's README)

## Project Objective (Reproducibility and Enhancement)

The primary goal is to construct an effective binary classification model to predict product recommendations through a highly reproducible and transparent data science workflow, while implementing methodological improvements over the original project. This involves:
* Detailed Exploratory Data Analysis (EDA) and visualization.
* Advanced data preprocessing and diverse feature engineering.
* Systematic feature selection from a large feature space.
* Rigorous building, training, and hyperparameter tuning of a Random Forest model.
* Comprehensive model evaluation.

The entire process is designed for easy reproducibility, with steps clearly documented in Jupyter notebooks (`DP_FE2.ipynb` and `FS_Model.ipynb`) and intermediate datasets saved.

## Methodology (Reproducible and Improved Workflow)

This project follows a structured and reproducible data science pipeline, divided into two main stages implemented in separate Jupyter notebooks:

1.  **Stage 1: EDA, Data Preprocessing, and Feature Engineering**
    * This stage begins with extensive Exploratory Data Analysis (EDA) to gain insights into the "Womens Clothing E-Commerce Reviews.csv" dataset.
    * Comprehensive data preprocessing steps are applied, including handling missing values (dropping rows with missing `Review Text` or `Recommended IND`, imputing `Title` with an empty string, removing duplicates, standardizing column names (lowercase, removing spaces), and detailed cleaning of the `Review Text` (lemmatization using NLTK's WordNetLemmatizer, stop word removal) to create the `cleaned_review_text` column.
    * Important data filtering is performed, such as removing inconsistent entries between `Recommended IND` and `Rating`  and limiting reviewer age (removing those > 80 years old). Outlier handling is applied to `Positive Feedback Count` using IQR capping, resulting in `positive_feedback_count_capped`.
    * New features are engineered: `review_length` (from `cleaned_review_text`), `sentiment_score` (using VADER on `cleaned_review_text`), One-Hot Encoding for categorical variables (`Division Name`, `Department Name`, `Class Name` with `drop_first=True`), and TF-IDF features (5000 features) from `cleaned_review_text`.
    * This stage concludes by saving the processed training (`FE_reviews_train.csv` - 15,454 records, 37 columns) and testing (`FE_reviews_test.csv` - 3,874 records, 37 columns) datasets, which serve as inputs for the next stage.

2.  **Stage 2: Feature Selection, Model Building, and Tuning**
    * This stage utilizes the processed data from Stage 1. Numerical features (`age`, `positivefeedbackcount`, `review_length`, `sentiment_score`) are scaled using `StandardScaler`.
    * Systematic feature selection is performed:
        * Highly correlated structured features are removed (correlation threshold > 0.85), resulting in 30 structured features. These are 'department_name_Intimate', 'class_name_Dresses', and 'class_name_Trend'.
        * The remaining 30 structured features are combined with 5000 TF-IDF features.
        * Random Forest Feature Importance and Mutual Information are then used to select the top 300 features from this combined set for modeling.
    * A Random Forest Classifier is built. Its hyperparameters are systematically tuned using `GridSearchCV` (cv=3, scoring='roc_auc') to optimize performance. The optimized parameters are `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}`.
    * The final tuned model is thoroughly evaluated on the test set using Classification Report, ROC AUC score, and Confusion Matrix.

## Comparison with Original Project and Enhancements

This project not only reproduces the basic steps but also implements significant enhancements compared to the methodology described in the README of janhavi-giri's original project, leading to a more transparent workflow and potentially improved results.

**1. Methodological Differences:**

* **Data Preprocessing & Feature Engineering:**
    * **This Project:** Implements more detailed data cleaning (filtering inconsistent data, advanced text cleaning with lemmatization, specific outlier capping for `Positive Feedback Count`, and enriched feature engineering (explicit `review_length` , VADER `sentiment_score` ).
    * **Original Project:** Describes preprocessing at a higher level ("selecting input/target features, one-hot encoding"). Specifics on deep text cleaning, outlier treatment, or VADER sentiment scores for Project #1 are not detailed.
* **Feature Selection:**
    * **This Project:** Employs correlation-based removal for structured features, followed by selecting the top 300 features from a large combined set (structured + TF-IDF) using Random Forest Importance and Mutual Information.
    * **Original Project:** Mentions Pearson Correlation, p-value, LassoCV, and decision tree importance, with varying numbers of features kept/dropped.
* **Model Tuning:**
    * **This Project:** Performs systematic hyperparameter tuning for the Random Forest model using `GridSearchCV`.
    * **Original Project:** Mentions Random Forest with/without class weights but does not detail extensive hyperparameter tuning for RF in its results section (though listed as a "Next step").

**2. Results Comparison and Explanation:**

The tuned Random Forest model in this project demonstrates strong performance on its processed test set.

| Metric                     | Original Project (RF Project #1, class_weights balanced) | This Project (Tuned RF, Test Set) |
|----------------------------|-------------------------------------------------------|-----------------------------------|
| **Accuracy** | 0.9333                                                | **0.95**               |
| **ROC AUC** | Not directly reported for RF Project #1               | **0.9899**             |
| **F1-score (Class 0)** | Not individually reported                             | **0.87**               |
| **F1-score (Class 1)** | Not individually reported                             | **0.97**               |
| **Recall (Class 0)** | ~0.942 (estimated from CM)                            | **0.92**               |
| **Recall (Class 1)** | ~0.931 (estimated from CM)                            | **0.96**               |
| **Precision (Class 0)** | ~0.749 (estimated from CM)                            | **0.83**               |
| **Precision (Class 1)** | ~0.987 (estimated from CM)                            | **0.98**               |
| **Total Test Samples** | 7046 (inferred from original CM)                      | 3874                    |

*Note: Precision/Recall for the original project are estimated from its provided Confusion Matrix and may not be exact without per-class support figures.*

*Reasons for improved performance in this project likely include:*
* **Higher Quality Input Data:** More rigorous data cleaning and filtering (e.g., removing inconsistent recommendations) led to a cleaner dataset for modeling.
* **Richer Feature Set:** The addition of features like `sentiment_score`  and `review_length`, along with robust text vectorization, provided more predictive signals.
* **Systematic Hyperparameter Optimization:** `GridSearchCV` ensured the Random Forest model was configured with optimal hyperparameters for this specific dataset.
* **Focus on ROC AUC:** Using ROC AUC for optimization is well-suited for potentially imbalanced classification tasks.

While the test set size in this project (3,874 samples ) is smaller than that inferred from the original project's CM (7,046 samples), this is a result of the stricter data filtering applied here. The percentage-based metrics (Accuracy, ROC AUC, F1-score) still indicate very strong performance on this project's refined dataset.

## Project Structure (For Reproducibility)

* **Sequential Notebooks:**
    * `DP_FE.ipynb`: Handles all data cleaning, EDA, text preprocessing, and feature engineering. It outputs intermediate CSV files.
    * `FS_Model.ipynb`: Uses the output from the first notebook for feature selection, model building, tuning, and evaluation.
* **Intermediate Data:** Saving `FE_reviews_train.csv` and `FE_reviews_test.csv`  allows for modularity and re-running parts of the pipeline independently.

## Results (This Project)

1.  **Feature Selection:**
    * 3 structured features dropped due to high correlation.
    * Top 300 features selected using Random Forest Importance for the final model.

2.  **Tuned Random Forest Model Performance (Test Set):**
    * **Best Hyperparameters:** `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300}`.
    * **ROC AUC Score:** 0.9899
    * **Accuracy:** 0.95
    * **Classification Report:**
        | Class             | Precision | Recall | F1-score | Support |
        |-------------------|-----------|--------|----------|---------|
        | 0 (Not Recommended) | 0.83      | 0.92   | 0.87     | 691     |
        | 1 (Recommended)   | 0.98      | 0.96   | 0.97     | 3183    |
    * **Confusion Matrix (Test Set):**
        * True Negatives (TN): 638
        * False Positives (FP): 53
        * False Negatives (FN): 131
        * True Positives (TP): 3052

## Conclusion

The refined Random Forest model, developed through a detailed and reproducible workflow involving advanced data cleaning, rich feature engineering, systematic feature selection, and rigorous hyperparameter tuning, demonstrates high efficacy in predicting product recommendations (Accuracy: 0.95, ROC AUC: 0.9899 on the test set). The model shows good generalization from the cross-validated training performance to the unseen test data. The improvements in methodology compared to the original project contribute to these robust results on the processed dataset.

## How to Reproduce Results

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/abdiyimer/Predictive-Analysis-Ecommerce](https://github.com/abdiyimer/Predictive-Analysis-Ecommerce)
    cd Predictive-Analysis-Ecommerce 
    ```

2.  **Set up Environment:**
    It is recommended to create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
    Install dependencies (ideally from a `requirements.txt` file you create):
    ```bash
    pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter
    # If you create a requirements.txt: pip install -r requirements.txt
    ```
    Download NLTK resources:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4') # For WordNet
    nltk.download('vader_lexicon') # For VADER sentiment analysis
    ```

3.  **Download Data:**
    Download the `Womens Clothing E-Commerce Reviews.csv` dataset from [Kaggle](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) and place it in a `Data/` subdirectory within your project folder.

4.  **Run Notebooks Sequentially:**
    * **Step 1:** Open and run all cells in `DP_FE.ipynb`. This will perform EDA, preprocessing, feature engineering, and save `Data/FE_reviews_train.csv` and `Data/FE_reviews_test.csv`.
    * **Step 2:** Open and run all cells in `FS_Model.ipynb`. This will load the processed data, perform feature selection, train, tune, and evaluate the model.

5.  **Verify Outputs:** The results, including model performance metrics and visualizations, will be displayed within the `FS_Model_1.ipynb` notebook and should align with those reported in this README.

---
