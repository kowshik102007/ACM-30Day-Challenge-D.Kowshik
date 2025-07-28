# Phase 4: SVD and PCA with TF-IDF on 20 Newsgroups Dataset

**Focus:** Text Preprocessing, TF-IDF Vectorization, Singular Value Decomposition (SVD), K-Means Clustering

## Tasks:
- **Data Cleaning**: Preprocessed raw text from the 20 Newsgroups dataset by removing headers, footers, and quotes for clean text analysis.
- **Feature Extraction**: Applied TF-IDF vectorization to convert text into numerical vectors, limiting vocabulary to 10,000 words to manage sparsity.
- **Dimensionality Reduction**: Used SVD to reduce the TF-IDF matrix to 2 components for visualization.
- **Clustering (Optional)**: Performed K-Means clustering with 20 clusters on the SVD-reduced data to assess alignment with the 20 newsgroup categories.
- **Model Evaluation**: Reported shapes of raw TF-IDF and SVD-reduced matrices, computed silhouette score for clustering, and visualized predicted vs. actual clusters with scatter plots and a confusion matrix.

## How I Solved It:
- **Setup Environment**: Imported libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (for `fetch_20newsgroups`, `TfidfVectorizer`, `TruncatedSVD`, `KMeans`, `silhouette_score`, `confusion_matrix`) for data handling, text processing, dimensionality reduction, clustering, and evaluation.
- **Data Loading**:
  - Loaded the 20 Newsgroups dataset using `fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))` from `scikit-learn`.
  - Dataset includes 18,846 documents across 20 categories (e.g., `alt.atheism`, `sci.med`, `rec.sport.baseball`).
  - Verified structure with `newsgroups.data` and `newsgroups.target`, confirming 18,846 samples and labels.
- **Data Cleaning**:
  - Removed headers, footers, and quotes during loading to eliminate metadata noise.
  - Used `TfidfVectorizer` with `stop_words='english'` to exclude common words, focusing on meaningful terms.
- **Feature Extraction**:
  - Applied `TfidfVectorizer(max_features=10000, stop_words='english')` to create a TF-IDF matrix, capturing term importance with a 10,000-word vocabulary.
- **Dimensionality Reduction**:
  - Used `TruncatedSVD(n_components=2, random_state=42)` to reduce the TF-IDF matrix to 2 components for visualization.
- **Clustering**:
  - Applied `KMeans(n_clusters=20, random_state=42)` to cluster the 2-component SVD data into 20 groups, approximating newsgroup categories.
- **Model Evaluation**:
  - Calculated silhouette score to evaluate clustering quality.
- **Visualization**:
  - Generated side-by-side scatter plots of the first two SVD components using `matplotlib`, colored by K-Means predicted clusters and actual categories with `tab20` colormap.

## 📊 Dataset Used
- **20 Newsgroups Dataset** – Sourced from `scikit-learn`’s `fetch_20newsgroups` (also available on [Kaggle](https://www.kaggle.com/datasets/crawford/20-newsgroups)).
  - Contains 18,846 text documents across 20 newsgroup categories, covering topics like technology, sports, and politics.
  - Each document is a labeled newsgroup post.

## 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Cleaning            | `fetch_20newsgroups`                      |
| Feature Extraction       | `TfidfVectorizer`                         |
| Dimensionality Reduction | `TruncatedSVD`                            |
| Clustering               | `KMeans`                                  |
| Evaluation               | `silhouette_score`, `confusion_matrix`     |
| Visualization            | `matplotlib`, `seaborn`                    |

## Files:
- `Phase_04.ipynb`: Jupyter notebook with code, scatter plots, and results for TF-IDF, SVD, and clustering.

## Explanation:
Text data, like the 20 Newsgroups dataset, is often transformed into a TF-IDF matrix, which is sparse (mostly zeros) due to the large number of unique words, making it hard to process efficiently. Dimensionality reduction, such as SVD or PCA, shrinks this large matrix into a smaller set of numbers (components) that still capture the main ideas or patterns in the data, like key topics in the texts. This makes it easier to analyze and visualize. Clustering, like K-Means, groups similar documents together without knowing their true labels, helping uncover natural patterns or themes in the data, such as grouping posts about sports or technology, even if the groups don’t perfectly match the original categories.
