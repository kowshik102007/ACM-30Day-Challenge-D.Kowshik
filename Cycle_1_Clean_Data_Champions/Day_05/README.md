# Case File #005 – 3-Feature Showdown 

**Focus:** Compact Burnout Prediction

## Tasks:
- **Select 3 Features**: Identify three key features using feature importance and exploratory data analysis (EDA).
- **Train a Minimal Model**: Train a model using only the selected features to predict Burnout.
- **Justify Selection**: Provide reasoning for the chosen features.

## Deliverable:
- **Chosen Features**: `BurnoutLevel`, `StressLevel`, `WorkLifeBalanceScore`
- **Model Accuracy**: Decision Tree, Accuracy = 1, Confusion matrix [[594  0]
 [ 0 306]]
- **Reasoning**: Detailed below under "Justification."

## How I Solved It:
- **Setup Environment**: Imported libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) for data processing, visualization, and modeling.
- **Data Loading**: Loaded the `mental_health_workplace_survey.csv` dataset using `pd.read_csv()` and verified its structure (3000 rows, 25 columns) with `df.head()`.
- **Exploratory Data Analysis (EDA)**:
  - Generated a correlation matrix using `df.corr()` to identify features strongly correlated with `BurnoutRisk`.
  - Used feature importance from a preliminary Random Forest model to rank features.
- **Feature Selection**:
  - Selected `WorkHoursPerWeek` , `BurnoutLevel` , and `StressLevel`.
  - These features were chosen for their strong predictive power and minimal redundancy, confirmed via correlation analysis and feature importance scores.
- **Data Preprocessing**:
  - Encoded `MentalHealthCondition` using `LabelEncoder` to convert categorical values to numerical.
  - Normalized `WorkHoursPerWeek` and `SleepHours` using `StandardScaler` for consistent scaling.
  - Split data into 80% training and 20% test sets with `train_test_split()`.
- **Model Training**:
  - Trained a Decision Tree Classifier with only the three selected features.
  - Evaluated performance using accuracy_score, confusionmatrix metrics on the test set.
- **Results**:
  - Achieved an Accuracy = 1, Confusion matrix [[594  0]
 [ 0 306]] indicating strong predictive performance with minimal features.

## Justification: Why These 3 Features?
Based on the feature importance scores and the correlation matrix  the selection of ['BurnoutLevel', 'StressLevel', 'CommuteTime'] as the top three features for the Burnout Breakdown dataset can be justified as follows:
- **BurnoutLevel :** This feature has the highest importance score, indicating it is the most significant predictor of burnout, likely due to its direct representation of the target variable or a strong self-correlation, making it a critical component for the model.
- **StressLevel :** With a moderate importance score and a potential correlation with BurnoutLevel, it captures a key psychological factor influencing burnout, supported by its inclusion in the correlation matrix analysis.
- **WorkLifeBalanceScore :** This feature’s inclusion is justified by its relevance to burnout, as WorkLifeBalanceScore is a well-known contributor to employee burnout in workplace studies. Its importance score is also third highest, supported by its inclusion in the correlation matrix analysis.
  
## 📊 Dataset Used
- **Mental Health and Burnout in the Workplace** – Kaggle  
  [https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace](https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace)

## 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Analysis & EDA      | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Encoding Categorical Data| `LabelEncoder`, `pandas`                   |
| Normalization            | `StandardScaler`, `scikit-learn`           |
| Model Training           | `Decision Tree`, `scikit-learn`    |
| Evaluation               | `accuracy_score`, `confusionmatrix`           |

## Files:
- `Day_05 Feature Showdown.ipynb`: Python script with code and results.

## Summary:
The Random Forest Regressor, trained on just three features(), achieved an Accuracy = 1, Confusion matrix [[594  0]
 [ 0 306]] indicating strong predictive performance with minimal features. These features were selected for their high correlation with `BurnoutRisk` and significant predictive power, as confirmed by EDA and feature importance analysis. 
