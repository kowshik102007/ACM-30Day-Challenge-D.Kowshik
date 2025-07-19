### Case File #002 – Feature Forge + Regression (Day 2)
**Focus:** Feature Engineering

#### Tasks:
- **Encoding Categorical Features**: Applied Label Encoding to categorical columns to reduce dimensionality, opting for this over One-Hot Encoding due to the significant increase in columns (from 25 to 54 with One-Hot vs. 25 with Label Encoding).
- **Normalization**: Used StandardScaler to normalize numerical features, ensuring consistent scales across the dataset for better model performance.
- **Feature Selection**: Utilized correlation analysis to select features with a correlation coefficient greater than 0.01 with `StressLevel`, excluding the target itself.
- **Interaction Features**: Created two interaction features: `Stress_WorkHours` (product of `StressLevel` and `WorkHoursPerWeek`) and `Sleep_Stress` (ratio of `SleepHours` to `StressLevel` with a small constant to avoid division by zero).
- **Model Training**: Trained Linear, Ridge, and Lasso regression models on the prepared data, evaluating them with MSE and R² metrics.


#### How I Solved It:
- **Setup Environment**: I started by importing key libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, and scikit-learn modules for preprocessing, modeling, and evaluation.
- **Data Loading**: Loaded the `mental_health_workplace_survey.csv` file using `pd.read_csv()` and inspected the first five rows with `df.head()` to confirm the structure (3000 rows, 25 columns).
- **Encoding Process**: Identified categorical columns using `df.select_dtypes()` and applied `LabelEncoder()` in a loop to transform them, choosing this method to maintain a compact dataset size compared to One-Hot Encoding.
- **Normalization**: Initialized `StandardScaler()`, fitted it on the training set, and transformed both training and test sets to ensure zero mean and unit variance.
- **Feature Selection**: Computed the correlation matrix with `df.corr()` focusing on `StressLevel`, filtered features with `abs(corr) > 0.01`, and excluded the target variable.
- **Interaction Features**: Engineered `Stress_WorkHours` and `Sleep_Stress` by multiplying and dividing respective columns, adding these to the selected features list.
- **Train-Test Split**: Split the data into 80% training and 20% test sets using `train_test_split()` with a random state for reproducibility.
- **Model Training**: Defined a dictionary of models (Linear, Ridge, Lasso), iterated over them, fitted each to the training data, predicted on the test set, and calculated MSE and R² using scikit-learn’s metrics.

## 📊 Dataset Used

- **Mental Health and Burnout in the Workplace** – Kaggle  
  [https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace](https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace)

---
### 🧰 Tools & Libraries Used

| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Cleaning            | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Missing Value Handling   | `pandas`                                   |
| Outlier Detection        | `IQR`, `matplotlib`, `seaborn`             |
| Encoding Categorical Data| `LabelEncoder`, `pandas`                   |
---

#### Files:
- `Day_02_Feature_Forge.ipynb`: Jupyter notebook with code and results.
- `Summary`: **Model Training + Scores**:
  - Linear: MSE = 0.83, R² = 0.88
  - Ridge: MSE = 0.83, R² = 0.88
  - Lasso: MSE = 2.17, R² = 0.70
  - For above mental_health_workplace_survey.csv the Linear model is the best, with an MSE of 0.83 and an R² of 0.88, and it’s tied with the Ridge model, which also got an MSE of 0.83 and an R² of 0.88. The Lasso model didn’t do as well, with an MSE of 2.17 and an R² of 0.70. Both Linear and Ridge are super good at predicting since they have the same low error and high R², meaning they match the data nicely. Linear takes the win because it’s simpler and skips the extra tweak Ridge adds, which didn’t help here. Lasso fell behind because it tries to zero out some features, which can leave out important info and mess up its predictions, leading to a higher MSE and lower R².

---
