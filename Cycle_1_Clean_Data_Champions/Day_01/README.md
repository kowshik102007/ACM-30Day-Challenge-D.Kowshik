
### Case File #001 – Burnout Breakdown (Day 1)
**Focus**: Data Cleaning + Exploratory Data Analysis (EDA)

#### Tasks:
- **Handle Missing Values**: Checked for missing values using `df.isnull().sum()`. The dataset had no missing values, so no imputation was needed. For robustness, I added a fallback to fill numerical columns with median and categorical columns with mode if any gaps appeared later.
- **Fix Outliers**: Identified potential outliers in `HoursPerDay` (derived from `WorkHoursPerWeek / 7`), `SleepHours`, and `StressLevel` using the IQR method. No outliers were removed as all values fell within the acceptable range (e.g., <18 hr workdays), but the process was tested and logged.
- **Visualize Relationships**: Created a scatter plot of `SleepHours` vs `StressLevel`, colored by `BurnoutRisk`, using `seaborn`. Insights: Lower sleep hours (e.g., <6) and higher stress levels (e.g., >6) correlate with increased burnout risk (BurnoutRisk = 1).


#### How I Solved It:
- **Initial Setup**: I began by importing the necessary libraries—`pandas` for data manipulation, `numpy` for numerical operations, and `matplotlib` and `seaborn` for visualization—into my Jupyter notebook environment. This set the foundation for a smooth workflow.
- **Data Loading and Exploration**: I loaded the `mental_health_workplace_survey.csv` dataset using `pd.read_csv(file)` and used `df.head()` to get a quick look at the first five rows, followed by `df.shape` and `df.info()` to understand the dataset's dimensions (3000 rows, 25 columns) and data types (mix of integers, floats, and objects).
- **Missing Value Check**: To ensure data integrity, I ran `df.isnull().sum()` to confirm there were no missing values across all 25 columns. As a proactive step, I implemented a fallback strategy using a loop to fill numerical columns with their median and categorical columns with their mode using `df.fillna()`, though it wasn’t needed this time.
- **Outlier Detection and Handling**: I calculated `HoursPerDay` by dividing `WorkHoursPerWeek` by 7 to convert weekly hours into a daily context, which helped identify potential work hour extremes. I then defined a custom `remove_outliers_IQR` function to apply the Interquartile Range (IQR) method on `HoursPerDay`, `SleepHours`, and `StressLevel`. This function calculated Q1, Q3, and the IQR, setting bounds to filter out values beyond 1.5 * IQR. I iterated over the features, applied the function, and printed the number of outliers removed (0 in this case), ensuring the dataset remained intact but ready for future adjustments.
- **Visualization Creation**: For the EDA, I used `seaborn.scatterplot()` to plot `SleepHours` against `StressLevel`, coloring points by `BurnoutRisk` with a `coolwarm` palette. This revealed a clear pattern where lower sleep and higher stress aligned with burnout risk, providing an early insight into employee well-being trends.

#### Files:
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

#### Dataset Used:
- `Day_01_Burnout_Breakdown.ipynb`: Jupyter notebook with code and visualizations.
- `mental_health_workplace_survey.csv`: Cleaned dataset

---


