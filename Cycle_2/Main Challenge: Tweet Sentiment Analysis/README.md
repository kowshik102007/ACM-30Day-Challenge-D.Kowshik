# Main Challenge – Tweet Sentiment Analysis

**Focus:** Text Preprocessing and Sentiment Classification

## Tasks:
- **Data Cleaning**: Processed raw tweet text by removing URLs, mentions, hashtags (keeping the word), special characters, and converting text to lowercase to ensure consistency for analysis.
- **Label Mapping**: Converted numeric sentiment labels (0, 4) into meaningful categories: Negative (0) and Positive (4), simplifying the classification task.
- **Feature Extraction**: Applied TF-IDF vectorization to transform tweet text into numerical vectors, limiting vocabulary to the top 5,000 words to reduce noise and prevent overfitting.
- **Model Training**: Trained an XGBoost classifier to predict tweet sentiment using the extracted features and sentiment labels.
- **Model Evaluation**: Evaluated the model using accuracy, a confusion matrix, and a classification report to assess precision, recall, and F1-score for each sentiment class.

## How I Solved It:
- **Setup Environment**: Imported essential libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (for preprocessing, train-test split, and evaluation), `re`, `string`, and `xgboost` for data manipulation, text cleaning, visualization, and modeling.
- **Data Loading**: Loaded the `sentiments.csv` file using `pd.read_csv()` with `ISO-8859-1` encoding and specified column names (`target`, `ids`, `date`, `flag`, `user`, `tweet`). Inspected the first five rows with `df.head()` to verify the dataset structure (1,600,000 rows, 6 columns).
- **Data Cleaning**:
  - Checked for duplicates using `df.duplicated().sum()`, confirming no duplicates.
  - Checked for missing values with `df.isna().sum()`, confirming no null values.
  - Created a `clean_tweet` function to preprocess tweet text:
    - Converted text to lowercase.
    - Removed URLs, mentions, hashtags (retaining words), punctuation, numbers, and extra whitespace using regular expressions (`re`) and `string.punctuation`.
    - Applied the function to the `tweet` column using `df['tweet'].apply(clean_tweet)`.
- **Label Mapping**:
  - Inspected `target` column with `df['target'].value_counts()`, revealing 800,000 Negative (0) and 800,000 Positive (4) labels, indicating a balanced dataset.
  - Kept labels as 0 and 4 for modeling, as no Neutral (2) labels were present, simplifying to a binary classification task.
- **Feature Extraction**:
  - Used `TfidfVectorizer` to convert cleaned tweet text into numerical vectors, emphasizing word importance while limiting vocabulary to 5,000 features to manage dimensionality.
  - Split the dataset into 80% training and 20% testing sets using `train_test_split()` with a random state of 42 for reproducibility.
- **Model Training**:
  - Selected XGBoost (`XGBClassifier`) due to its robustness in handling text-based features, efficiency with large datasets, and ability to capture complex patterns through boosting.
  - Trained the model on the training set with `random_state=42`, `use_label_encoder=False`, and `eval_metric='mlogloss'` to ensure compatibility and proper evaluation.
- **Model Evaluation**:
  - Predicted sentiments on the test set and calculated accuracy using `accuracy_score`, achieving 0.76.
  - Generated a confusion matrix using `confusion_matrix` and visualized it with a `seaborn` heatmap to show prediction performance for Negative and Positive classes.
  - Produced a classification report using `classification_report`, detailing:
    - **Negative**: Precision (0.79), Recall (0.71), F1-score (0.75)
    - **Positive**: Precision (0.74), Recall (0.81), F1-score (0.77)
    - **Overall Accuracy**: 0.76, with balanced macro and weighted averages.

## 📊 Dataset Used
- **Sentiment140 Dataset** – [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

  
## 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Cleaning            | `pandas`, `numpy`, `re`, `string`          |
| EDA and Visualization    | `matplotlib`, `seaborn`                    |
| Feature Extraction       | `TfidfVectorizer`                         |
| Train-Test Split         | `train_test_split`                        |
| Model Training           | `XGBClassifier`                           |
| Evaluation               | `accuracy_score`, `confusion_matrix`, `classification_report` |

## Challenges and Solutions:
- **Challenge**: Handling noisy tweet text (URLs, mentions, hashtags, etc.).  
  **Solution**: Developed a robust `clean_tweet` function using regular expressions to systematically remove noise while preserving meaningful content (e.g., hashtag words).
- **Challenge**: Large dataset size (1.6 million rows) causing computational constraints.  
  **Solution**: Limited TF-IDF vocabulary to 5,000 features to reduce dimensionality and used XGBoost for efficient processing of large-scale text data.

## Files:
- `Tweet Sentiment Analysis.ipynb`: Jupyter notebook containing the complete code, visualizations, and results.

## Summary:
- **Model Choice**: Chose XGBoost over other classification algorithms for sentiment analysis due to this advantages:

1. **Handles High-Dimensional Sparse Data**:
   - Sentiment analysis often uses TF-IDF or word embeddings, creating sparse, high-dimensional data. XGBoost’s sparsity-aware algorithm efficiently processes these, reducing memory use and speeding up computation compared to algorithms like SVM, which struggle with scalability.

2. **Robust to Noisy Text**:
   - Text data in sentiment analysis (e.g., reviews with slang or typos) is noisy. XGBoost’s iterative boosting focuses on correcting misclassified samples, and its L1/L2 regularization prevents overfitting, unlike Logistic Regression or Naive Bayes, which are more noise-sensitive.

3. **Fast and Scalable**:
   -  XGBoost’s parallelized tree construction and GPU support make it faster for large sentiment datasets compared to SVM’s high computational cost. It balances speed and accuracy better than Random Forest, which is slower for large-scale text data.
     
- **TF-IDF Role**: TF-IDF effectively captured word importance by weighting terms based on their frequency and rarity across tweets, enabling the model to focus on sentiment-relevant words while reducing noise from common or irrelevant terms.
- **Model Performance**:
  - **Accuracy**: 0.76
  - **Negative Class**: Precision (0.79), Recall (0.71), F1-score (0.75)
  - **Positive Class**: Precision (0.74), Recall (0.81), F1-score (0.77)
- The model achieved a balanced performance, with slightly higher recall for Positive tweets (0.81), indicating better detection of positive sentiment, likely due to distinct positive language patterns captured by TF-IDF. The confusion matrix visualization highlighted areas for improvement, such as reducing false negatives for the Negative class.
