### Case File #003 – Classifier Arena (Day 3)
**Focus:** Building and Evaluating Classifiers

#### Tasks:
- **Encoding Categorical Features**: Applied LabelEncoder to categorical columns to convert them into numerical format, choosing this over One-Hot Encoding to avoid a significant increase in dimensionality.
- **Train-Test Split**: Divided the dataset into 80% training and 20% test sets with `BurnoutRisk` as the target variable, using a random state for reproducibility.
- **Standardization**: Used StandardScaler to normalize numerical features, ensuring consistent scales for improved model performance.
- **Model Training**: Trained Logistic Regression and Linear Discriminant Analysis (LDA) models on the prepared data.
- **Evaluation**: Assessed models using Accuracy, Confusion Matrix, and ROC-AUC, including plotting ROC curves.

#### How I Solved It:
- **Setup Environment**: Imported key libraries including `pandas`, `numpy`, `matplotlib`, and scikit-learn modules for data handling, preprocessing, modeling, and visualization.
- **Data Loading**: Loaded the `mental_health_workplace_survey.csv` file with `pd.read_csv()` and checked the first five rows using `df.head()` to verify the structure (3000 rows, 25 columns).
- **Encoding Process**: Identified categorical columns with `df.select_dtypes()` and applied `LabelEncoder()` in a loop to transform them, maintaining a compact dataset.
- **Train-Test Split**: Split the data into training and test sets using `train_test_split()` with a 0.2 test size and random state 5.
- **Standardization**: Initialized `StandardScaler()`, fitted it on the training set, and transformed both training and test sets for zero mean and unit variance.
- **Model Training**: Defined a dictionary of models (Logistic Regression, LDA), fitted them to the training data, and predicted on the test set.
- **Evaluation**: Calculated Accuracy, ROC-AUC, and Confusion Matrix using scikit-learn metrics, and plotted ROC curves with `matplotlib`.

## 📊 Dataset Used
- **Mental Health and Burnout in the Workplace** – Kaggle  
  [https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace](https://www.kaggle.com/datasets/khushikyad001/mental-health-and-burnout-in-the-workplace)

---
### 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Cleaning            | `pandas`, `numpy`, `matplotlib`            |
| Encoding Categorical Data| `LabelEncoder`, `pandas`                   |
| Model Training           | `LogisticRegression`, `LinearDiscriminantAnalysis` |
| Evaluation               | `accuracy_score`, `roc_auc_score`, `confusion_matrix`, `roc_curve`|

#### Files:
- `Day_03_Classifier_Arena.ipynb`: Jupyter notebook with code and results.

#### Summary:
- **Model Training + Scores:**
  - Logistic Regression: Accuracy = 0.99, ROC-AUC = 1.00, Confusion Matrix = [[408   4], [  1 187]]
  - LDA: Accuracy = 0.96, ROC-AUC = 1.00, Confusion Matrix = [[390  22], [  0 188]]
- For the mental_health_workplace_survey.csv, the Logistic Regression model performed better with an Accuracy of 0.99 and an ROC-AUC of 1.00, compared to LDA’s 0.96 Accuracy and 1.00 ROC-AUC. Both models achieved perfect ROC-AUC scores, indicating excellent separation of burnout and no burnout classes. However, Logistic Regression’s higher accuracy (0.99 vs. 0.96) and fewer errors (5 total, with 4 false positives and 1 false negative) versus LDA’s 22 errors (all false positives) give it the edge. Logistic’s flexibility in fitting the decision boundary likely aligns better with the dataset’s patterns, while LDA’s equal variance assumption may lead to more misclassifications.

---
