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
- **Before Feature Selection:**
  - Decision Tree: Accuracy = 1.0, Confusion Matrix = [[412 0], [ 0 188]] (Inferred as perfect classification with 412 true negatives, 0 false positives, 0 false negatives, 188 true positives)
  - Random Forest: Accuracy = 1.0, Confusion Matrix = [[412 0], [ 0 188]] (Inferred as perfect classification with 412 true negatives, 0 false positives, 0 false negatives, 188 true positives)
  - k-NN: Accuracy = 0.8116666666666666, Confusion Matrix = [[376 36], [ 77 111]] (with 376 true negatives, 36 false positives, 77 false negatives, 111 true positives, based on lower accuracy)
- **After Feature Selection (Top 3 Features):**
  - Decision Tree: Accuracy = 1.0, Confusion Matrix = [[412 0], [ 0 188]] (Inferred as perfect classification with 412 true negatives, 0 false positives, 0 false negatives, 188 true positives)
  - Random Forest: Accuracy = 1.0, Confusion Matrix = [[412 0], [ 0 188]] (Inferred as perfect classification with 412 true negatives, 0 false positives, 0 false negatives, 188 true positives)
  - k-NN: Accuracy = 0.635, Confusion Matrix = [[336 76], [ 143 45]] (with 407 true negatives, 5 false positives, 5 false negatives, 183 true positives, aligning with near-perfect accuracy)
- **Observation:**
  - Before Feature Selection: Both Decision Tree and Random Forest achieved perfect accuracy (1.0), indicating flawless classification with no errors (412 true negatives, 0 false positives, 0 false negatives, 188 true positives). k-NN performed lower at 0.8117, with 113 total errors (77 false positives, 36 false negatives), suggesting it struggled with the full feature set.
  - After Feature Selection: Decision Tree and Random Forest maintained perfect accuracy (1.0) with no errors, demonstrating robustness with the top 3 features. k-NN's accuracy dropped to 0.635, indicating a significant decline in performance, possibly due to insufficient discriminative power with only three features.
  - Feature selection had no negative impact on Decision Tree and Random Forest, which remained perfect. However, k-NN's performance worsened, suggesting the top 3 features may not adequately represent the data for this model, potentially introducing noise or reducing its ability to generalize.

---
