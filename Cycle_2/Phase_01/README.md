# Phase 1: Bagging and Boosting

**Focus:** Ensemble Learning with Bagging and Boosting

## Tasks:
- **Train Models**:
  - RandomForestClassifier
  - AdaBoostClassifier
  - XGBoostClassifier
- **Compare Accuracy**: Evaluate and compare the performance of the three models using accuracy, confusion matrix, and classification report metrics.

## How I Solved It:
- **Setup Environment**: Imported key libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`) for data processing, visualization, and modeling.
- **Data Loading**: Loaded the `breast-cancer.csv` dataset using `pd.read_csv()` and inspected the first five rows with `df.head()` to confirm the structure (569 rows, 32 columns).
- **Data Preprocessing**:
  - Checked for missing values using `df.isnull().sum()`, confirming no missing data.
  - Encoded the categorical `diagnosis` column (M = malignant, B = benign) using `LabelEncoder` to convert to numerical values (M → 1, B → 0).
  - Dropped the `id` column as it is irrelevant for prediction.
- **Train-Test Split**: Split the data into 80% training and 20% test sets using `train_test_split()`.
- **Normalization**: Applied `StandardScaler` to standardize numerical features, fitting on the training set and transforming both training and test sets for consistent scaling.
- **Model Training**: Defined a dictionary of models (RandomForestClassifier, AdaBoostClassifier, XGBoostClassifier) with 100 estimators each and a fixed random state for reproducibility.
- **Model Evaluation**: Trained each model, predicted on the test set, and calculated:
  - Accuracy using `accuracy_score`.
  - Confusion matrix using `confusion_matrix`.
  - Classification report (precision, recall, F1-score) using `classification_report` for both Benign and Malignant classes.

## 📊 Dataset Used
- **Breast Cancer Dataset** – Kaggle  
  [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

## 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Loading & Cleaning  | `pandas`, `numpy`                         |
| Visualization            | `matplotlib`, `seaborn`                   |
| Encoding Categorical Data| `LabelEncoder`, `scikit-learn`            |
| Normalization            | `StandardScaler`, `scikit-learn`          |
| Model Training           | `RandomForestClassifier`, `AdaBoostClassifier`, `XGBClassifier`, `scikit-learn`, `xgboost` |
| Evaluation               | `accuracy_score`, `confusion_matrix`, `classification_report`, `scikit-learn` |

## Files:
- `Phase_01_Bagging_and_Boosting_concepts.ipynb`: Jupyter notebook with code and results.

## Results:
| Model           | Accuracy | Benign Precision | Benign Recall | Benign F1-Score | Malignant Precision | Malignant Recall | Malignant F1-Score | Misclassifications |
|-----------------|----------|------------------|---------------|-----------------|---------------------|------------------|--------------------|--------------------|
| Random Forest   | 0.965    | 0.96             | 0.98          | 0.97            | 0.98                | 0.94             | 0.96               | 4                  |
| AdaBoost        | 0.982    | 0.97             | 1.00          | 0.99            | 1.00                | 0.96             | 0.98               | 2                  |
| XGBoost         | 0.974    | 0.96             | 1.00          | 0.98            | 1.00                | 0.94             | 0.97               | 3                  |

**Confusion Matrices**:
- **Random Forest**: [[65, 1], [3, 45]]
- **AdaBoost**: [[66, 0], [2, 46]]
- **XGBoost**: [[66, 0], [3, 45]]
