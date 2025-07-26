# Phase 2: Support Vector Machines (SVM) for Credit Card Fraud Detection

**Focus:** Margin, Kernels, Linear/Non-Linear Classification, Dimensionality Reduction using PCA

## Tasks:
- **Data Loading and Preprocessing**: Loaded the `creditcard.csv` dataset, inspected for missing values and duplicates, and standardized features to ensure consistent scaling for SVM.
- **Dimensionality Reduction**: Applied Principal Component Analysis (PCA) to reduce the 30-dimensional feature space to 2D for visualization, preserving key variance in the data.
- **Model Training**: Trained Support Vector Machine (SVM) classifiers with three different kernels (Linear, RBF, Polynomial) to classify transactions as non-fraudulent (0) or fraudulent (1).
- **Model Evaluation**: Evaluated each model using accuracy, confusion matrix, and classification report to assess performance on the imbalanced dataset.
- **Visualization**: Visualized decision boundaries for each kernel using a 2D PCA plot to illustrate how different kernels affect classification.

## How I Solved It:
- **Setup Environment**: Imported necessary libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (for preprocessing, PCA, SVM, train-test split, and evaluation metrics), and `DecisionBoundaryDisplay` for visualization.
- **Data Loading**:
  - Loaded `creditcard.csv` using `pd.read_csv()`, containing 284,807 transactions with 30 features (`Time`, `V1`–`V28`, `Amount`) and a target (`Class`: 0 for non-fraud, 1 for fraud).
  - Inspected with `df.head()` and `df.info()`, confirming 31 columns (30 features + 1 target) and no missing values (`df.isnull().sum()`).
- **Data Preprocessing**:
  - Checked for duplicates with `df.duplicated().sum()`, finding none.
  - Separated features (`X`: all columns except `Class`) and target (`y`: `Class`).
  - Standardized features using `StandardScaler` to ensure zero mean and unit variance, critical for SVM performance.
  - Split data into 80% training and 20% testing sets using `train_test_split` with `random_state=42` for reproducibility.
- **Dimensionality Reduction**:
  - Applied PCA to reduce the 30 features to 2 components for visualization, capturing the most significant variance in the data.
- **Model Training**:
  - Trained SVM classifiers (`SVC`) with three kernels:
    - **Linear Kernel**: Assumes linear separability between classes.
    - **RBF Kernel**: Captures non-linear patterns using a Gaussian radial basis function.
    - **Polynomial Kernel**: Models non-linear relationships with polynomial decision boundaries.
  - Used `random_state=42` for reproducibility and default parameters for simplicity.
- **Model Evaluation**:
  - Evaluated each model on the test set:
    - **Accuracy**: Measured overall correctness.
    - **Confusion Matrix**: Analyzed true positives, false positives, true negatives, and false negatives.
    - **Classification Report**: Provided precision, recall, and F1-score, with `zero_division=0` to handle cases with no predicted samples (common for the minority class in imbalanced data).
- **Visualization**:
  - Plotted 2D PCA-transformed training data with decision boundaries for each kernel using `DecisionBoundaryDisplay`, highlighting how kernel choice affects the separation of classes.

## 📊 Dataset Used
- **Credit Card Fraud Detection Dataset** –  [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
  - Contains 284,807 transactions with 30 features (`Time`, `V1`–`V28`, `Amount`) and a binary target (`Class`: 0 = non-fraud, 1 = fraud).
  - Highly imbalanced: ~99.83% non-fraudulent (284,315) and ~0.17% fraudulent (492 transactions).

## 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Loading & Cleaning  | `pandas`, `numpy`                          |
| Preprocessing            | `StandardScaler`, `train_test_split`       |
| Dimensionality Reduction | `PCA`                                      |
| Model Training           | `SVC`                                      |
| Evaluation               | `accuracy_score`, `confusion_matrix`, `classification_report` |
| Visualization            | `matplotlib`, `seaborn`, `DecisionBoundaryDisplay` |

## Challenges and Solutions:
- **Challenge**: Imbalanced dataset with only 0.17% fraudulent transactions, risking poor model performance on the minority class.  
  **Solution**: Used classification report to focus on precision, recall, and F1-score for the minority class (fraud), as accuracy alone is misleading in imbalanced datasets.
- **Challenge**: High-dimensional data (30 features) complicating visualization and computation.  
  **Solution**: Applied PCA to reduce to 2D for visualization and standardized features to ensure SVM’s sensitivity to scale was addressed.

## Effect of Kernels:
- **Linear Kernel**: Assumes a linear decision boundary, which may struggle with complex, non-linear patterns in fraud data. Likely to have lower recall for the fraud class due to oversimplification.
- **RBF Kernel**: Uses a Gaussian function to create flexible, non-linear boundaries, potentially capturing intricate fraud patterns better but risking overfitting if not tuned properly.
- **Polynomial Kernel**: Models non-linear relationships with polynomial boundaries, offering a balance between flexibility and complexity but computationally intensive for high-degree polynomials.

## Files:
- `Phase_02.ipynb`: Jupyter notebook containing the complete code, visualizations, and results.

## Performance:
Both Linear and RBF Kernel SVM models achieved identical performance metrics, highlighting challenges with the imbalanced dataset. The high accuracy (0.9983) is misleading due to the dominance of the non-fraudulent class (56,864 samples vs. 98 fraudulent samples in the test set). Both models failed to detect any fraudulent transactions, as evidenced by zero precision, recall, and F1-score for the fraud class.

### Performance Comparison Table
| Kernel   | Accuracy | Class | Precision | Recall | F1-Score | Support | Confusion Matrix  |
|----------|----------|-------|-----------|--------|----------|---------|-----------------------------------|
| Linear   | 0.9983   | 0     | 1.00      | 1.00   | 1.00     | 56,864  | [56864, 0, 98, 0]                |
|          |          | 1     | 0.00      | 0.00   | 0.00     | 98      |                                   |
| RBF      | 0.9983   | 0     | 1.00      | 1.00   | 1.00     | 56,864  | [56864, 0, 98, 0]                |
|          |          | 1     | 0.00      | 0.00   | 0.00     | 98      |                                   |

## Summary:
- **PCA Role**: Reduced 30 features to 2D for visualization, enabling insight into how kernels separate classes in a simplified space.
- **Kernel Impact**: 
  - Linear kernel provides simplicity but may miss non-linear fraud patterns.
  - RBF kernel offers flexibility for complex patterns but requires careful parameter tuning.
  - Polynomial kernel balances complexity but is computationally expensive.
- **Performance**: Both Linear and RBF kernels achieved 99.83% accuracy but failed to detect any fraudulent transactions (0 recall for class 1), emphasizing the need for handling class imbalance and tuning hyperparameters.
- **Visualization**: 2D PCA plots with decision boundaries illustrated how each kernel shapes the classification space, with RBF kernels likely showing more complex boundaries than Linear.

## Note: I didnot perform the SVM using polynomial kernelas it is computationally intensive for high-degree polynomials and taking more time.
