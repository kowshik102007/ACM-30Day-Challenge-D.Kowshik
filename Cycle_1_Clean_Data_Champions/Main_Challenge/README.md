### Main Challenge – Dirty Dataset Detective
**Focus:** Data Cleaning and Regression Modeling

#### Tasks:
- **Data Cleaning**: Addressed missing values, inconsistent or misspelled categories, outliers, and redundant features in the dataset to ensure data integrity.
- **Exploratory Data Analysis (EDA) and Feature Selection**: Conducted EDA to understand data distribution and correlations, followed by feature selection based on correlation analysis to identify key predictors of insurance charges.
- **Model Training**: Trained a linear regression model and improved it with polynomial regression to achieve an R² greater than 0.85.
- **Model Improvement**: Enhanced the model's performance by addressing non-linear relationships and outliers, increasing R² from below 0.6 to above 0.85.

#### How I Solved It:
- **Setup Environment**: Imported essential libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, and scikit-learn modules for data manipulation, visualization, preprocessing, and modeling.
- **Data Loading**: Loaded the `insurance.csv` file using `pd.read_csv()` and inspected the first five rows with `df.head()` to verify the structure (1338 rows, 7 columns).
- **Data Cleaning**: Checked for missing values with `df.isnull().sum()`, confirming no missing data. Ensured categorical columns (`sex`, `smoker`, `region`) were consistent, as no misspelling or inconsistencies were detected. Identified and handled outliers implicitly through standardization and polynomial modeling, with no redundant features requiring removal.
- **EDA Process**: Generated displots for numerical columns (`age`, `bmi`, `children`, `charges`) to analyze distributions, pie charts for categorical columns (`sex`, `smoker`, `region`) to assess proportions, and a correlation matrix heatmap to evaluate relationships with `charges`. Observations guided feature selection, highlighting `smoker`, `age`, and `bmi` as significant predictors.
- **Feature Selection**: Based on the correlation matrix, selected `age`, `bmi`, `children`, `sex`, `smoker`, and `region` as features, with `smoker` (correlation ~0.79) being the strongest predictor, followed by `age` (~0.30) and `bmi` (~0.20). `children`, `sex`, and `region` showed weak or negligible correlations (~0.07, ~0, ~0).
- **Train-Test Split**: Split the data into 70% training and 30% test sets using `train_test_split()` with a random state of 5 for reproducibility.
- **Standardization**: Applied `StandardScaler()` to normalize numerical features in both training and test sets, ensuring consistent scales for modeling.
- **Model Training**:
  - **Linear Regression**: Trained a linear model, achieving an MSE of 34,543,813.62 and an R² of 0.76.
  - **Polynomial Regression**: Used `PolynomialFeatures(degree=2)` to capture non-linear relationships, improving the model to an MSE of 21,049,918.50 and an R² of 0.85, meeting the challenge goal.
- **Improvement Strategy**: The shift to polynomial regression addressed non-linear patterns (e.g., exponential charge increases for smokers) and outliers, significantly boosting R² and reducing MSE.

## 📊 Dataset Used

- **Medical Insurance Cost Prediction** – Kaggle  
  [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---
### 🧰 Tools & Libraries Used

| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Cleaning            | `pandas`, `numpy`                          |
| EDA and Visualization    | `matplotlib`, `seaborn`                    |
| Encoding Categorical Data| `OneHotEncoding`, `pandas`   |
| Normalization            | `StandardScaler`                           |
| Model Training           | `LinearRegression`, `PolynomialFeatures`   |
| Evaluation               | `mean_squared_error`, `r2_score`           |

---

#### Files:
- `Dirty_Dataset_Detective.ipynb`: Jupyter notebook with code, visualizations, and results.

#### Summary: 
- **Model Training + Scores**:
  - Linear Regression: MSE = 34,543,813.62, R² = 0.76
  - Polynomial Regression: MSE = 21,049,918.50, R² = 0.85
- The R² score went up from 0.76 with linear regression to 0.85 with polynomial regression mainly because polynomial regression is better at handling curves and changes that aren’t straight lines, like the big jump in insurance costs for smokers. Linear regression can’t easily follow these ups and downs, but polynomial regression can adjust to them more smoothly. This helps it match the data more closely, especially with tricky patterns like higher charges for smokers, making it a more accurate way to understand the trends and explain more of what’s going on in the data.
---
