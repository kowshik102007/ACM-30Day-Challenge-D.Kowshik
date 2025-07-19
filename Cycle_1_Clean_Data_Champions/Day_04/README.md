### Case File #004 – Tree-Based Models + k-NN + Feature Selection (Day 4)
**Focus:** Advanced Classifiers and Feature Selection

#### Tasks:
- **Encoding Categorical Features**: Applied LabelEncoder to categorical columns to convert them into numerical format for modeling.
- **Train-Test Split**: Divided the dataset into 80% training and 20% test sets with `BurnoutRisk` as the target, using a random state for reproducibility.
- **Standardization**: Utilized StandardScaler to normalize numerical features for consistent scale across models.
- **Model Training**: Trained Decision Tree, Random Forest, and k-NN classifiers on the full dataset.
- **Feature Selection**: Used Random Forest feature importances to select the top 3 most important features and dropped weaker ones.
- **Retraining**: Trained the same models with the top 3 features and compared performance.

#### How I Solved It:
- **Setup Environment**: Imported libraries including `pandas`, `matplotlib`, `seaborn`, and scikit-learn for data handling, visualization, and modeling.
- **Data Loading**: Loaded the `mental_health_workplace_survey.csv` file and inspected the structure (3000 rows, 25 columns).
- **Encoding Process**: Identified categorical columns with `df.select_dtypes()` and transformed them using `LabelEncoder()` in a loop.
- **Train-Test Split**: Split data into training and test sets with `train_test_split()`, setting test_size=0.2 and random_state=5.
- **Standardization**: Applied `StandardScaler()` to fit and transform training and test sets.
- **Model Training**: Defined a function to train Decision Tree, Random Forest, and k-NN models, calculating accuracy and plotting confusion matrices.
- **Feature Selection**: Extracted feature importances from Random Forest, selected the top 3 features, and retrained models on the reduced dataset.
- **Comparison**: Evaluated and compared accuracy and confusion matrices before and after feature selection.

## 📊 Dataset Used
- **Mental Health and Burnout in the Workplace** – Kaggle  
  [https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace](https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace)

---
### 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Cleaning            | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Encoding Categorical Data| `LabelEncoder`, `pandas`                   |
| Model Training           | `DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier` |
| Evaluation               | `accuracy_score`, `confusion_matrix`       |

#### Files:
- `Day_04_Tree-Based_Models_k-NN_Feature_Selection.ipynb`: Jupyter notebook with code and results.

#### Summary:
- **Before Feature Selection (Full Dataset):**
  - Decision Tree: Accuracy = 0.98, Confusion Matrix = [[402   9], [  3 186]]
  - Random Forest: Accuracy = 0.99, Confusion Matrix = [[407   4], [  2 187]]
  - k-NN: Accuracy = 0.97, Confusion Matrix = [[398  13], [  4 185]]
- **After Feature Selection (Top 3 Features):**
  - Decision Tree: Accuracy = 0.96, Confusion Matrix = [[395  16], [  6 183]]
  - Random Forest: Accuracy = 0.97, Confusion Matrix = [[400  11], [  5 184]]
  - k-NN: Accuracy = 0.95, Confusion Matrix = [[392  19], [  7 182]]
- The full dataset models outperformed the 3-feature models, with Random Forest achieving the highest accuracy (0.99) before selection and 0.97 after. The slight drop post-selection suggests the top 3 features retain most predictive power but lose some granularity. Random Forest consistently performed best due to its ensemble nature, while k-NN was most affected by reduced features.

---
