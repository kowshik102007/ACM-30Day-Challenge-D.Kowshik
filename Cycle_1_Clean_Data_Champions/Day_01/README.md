
### Case File #001 – Burnout Breakdown (Day 1)
**Focus**: Data Cleaning + Exploratory Data Analysis (EDA)

#### Tasks:
- **Handle Missing Values**: Checked for missing values using `df.isnull().sum()`. The dataset had no missing values, so no imputation was needed. For robustness, I added a fallback to fill numerical columns with median and categorical columns with mode if any gaps appeared later.
- **Fix Outliers**: Identified potential outliers in `HoursPerDay` (derived from `WorkHoursPerWeek / 7`), `SleepHours`, and `StressLevel` using the IQR method. No outliers were removed as all values fell within the acceptable range (e.g., <18 hr workdays), but the process was tested and logged.
- **Visualize Relationships**: Created a scatter plot of `SleepHours` vs `StressLevel`, colored by `BurnoutRisk`, using `seaborn`. Insights: Lower sleep hours (e.g., <6) and higher stress levels (e.g., >6) correlate with increased burnout risk (BurnoutRisk = 1).


#### How I Solved It:
- Loaded the dataset with `pandas` and explored its shape (3000 rows, 25 columns) and data types using `df.info()`.
- Used `df.isnull().sum()` to confirm no missing values, with a safety net of median/mode imputation.
- Calculated `HoursPerDay` to check work hours and applied the IQR method to handle outliers, printing the number removed (0 in this case).
- Generated the scatter plot with `seaborn` to visualize relationships, adding labels and a legend for clarity.

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
- [`Day_01_Burnout_Breakdown.ipynb`](./Day_01_Burnout_Breakdown.ipynb): Jupyter notebook with code and visualizations.
- `mental_health_workplace_survey.csv`: Cleaned dataset

---


