# Phase 3: Unsupervised Learning for Iris Species Classification

**Focus:** Clustering (K-Means & Hierarchical), Dimensionality Reduction (PCA)

## Tasks:
- **Data Loading and Preprocessing**: Loaded the `Iris.csv` dataset, checked for missing values, dropped unnecessary columns, and standardized features for clustering.
- **Dimensionality Reduction**: Applied Principal Component Analysis (PCA) to reduce the 4-dimensional feature space to 2D for visualization and clustering.
- **Clustering**:
  - **K-Means Clustering**: Applied K-Means with the elbow method to determine the optimal number of clusters.
  - **Hierarchical Clustering**: Performed hierarchical clustering using Ward's linkage method and visualized the results with a dendrogram.
- **Evaluation**: Used silhouette scores to assess clustering quality for both methods.
- **Visualization**: Plotted 2D PCA-transformed data with cluster labels for K-Means and hierarchical clustering, and generated a dendrogram for hierarchical clustering.


## How I Solved It:
- **Setup Environment**: Imported necessary libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (for preprocessing, PCA, K-Means, train-test split, and silhouette score), and `scipy` (for hierarchical clustering and dendrogram).
- **Data Loading**:
  - Loaded `Iris.csv` using `pd.read_csv()`, containing 150 samples with 5 features (`Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) and a target (`Species`: Iris-setosa, Iris-versicolor, Iris-virginica).
  - Inspected with `df.head()` and `df.info()`, confirming 6 columns (5 features + 1 target) and no missing values (`df.isnull().sum()`).
- **Data Preprocessing**:
  - Dropped the `Id` column as it was irrelevant for clustering.
  - Encoded the categorical `Species` column using `LabelEncoder` for potential comparison with true labels, though not used in unsupervised learning.
  - Separated features (`X`: all columns except `Species`) and standardized them using `StandardScaler` to ensure zero mean and unit variance, critical for K-Means and hierarchical clustering.
- **Dimensionality Reduction**:
  - Applied PCA to reduce the 4 features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) to 2 principal components for visualization and clustering, capturing significant variance in the data.
- **Clustering**:
  - **K-Means Clustering**:
    - Applied K-Means with a range of cluster numbers (1 to 10) to compute inertia (within-cluster sum of squares).
    - Used the elbow method to identify the optimal number of clusters by plotting inertia vs. number of clusters, typically selecting 3 clusters for the Iris dataset (corresponding to the three species).
    - Trained K-Means with the optimal number of clusters and obtained cluster labels for the PCA-transformed data.
  - **Hierarchical Clustering**:
    - Performed hierarchical clustering on PCA-transformed data using Ward's linkage method (`linkage` from `scipy.cluster.hierarchy`).
    - Generated a dendrogram to visualize the hierarchical structure and determine the number of clusters by setting a threshold (3 clusters).
    - Assigned cluster labels using `fcluster` with `maxclust=3`.
- **Evaluation**:
  - Computed silhouette scores for both K-Means and hierarchical clustering to evaluate cluster cohesion and separation.
- **Visualization**:
  - Plotted 2D PCA-transformed data with K-Means cluster labels using `sns.scatterplot`, showing clusters in different colors.
  - Plotted 2D PCA-transformed data with hierarchical cluster labels using `sns.scatterplot` (as shown in the notebook output).
  - Generated a dendrogram for hierarchical clustering using `dendrogram` from `scipy.cluster.hierarchy` to visualize the merging process.

## 📊 Dataset Used
- **Iris Species Dataset** – Sourced from [Kaggle](https://www.kaggle.com/datasets/uciml/iris)  
  - Contains 150 samples with 4 features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) and a target (`Species`: Iris-setosa, Iris-versicolor, Iris-virginica).
  - Balanced dataset: 50 samples per species.

## 🧰 Tools & Libraries Used
| Task                     | Tools / Libraries                          |
|--------------------------|--------------------------------------------|
| Data Loading & Cleaning  | `pandas`, `numpy`                          |
| Preprocessing            | `StandardScaler`, `LabelEncoder`           |
| Dimensionality Reduction | `PCA`                                      |
| Clustering               | `KMeans`, `scipy.cluster.hierarchy` (linkage, dendrogram, fcluster) |
| Evaluation               | `silhouette_score`                         |
| Visualization            | `matplotlib`, `seaborn`                    |

## Performance:
Both K-Means and hierarchical clustering were evaluated using silhouette scores, which measure how similar an object is to its own cluster compared to other clusters (range: -1 to 1, higher is better). K-Means achieved a slightly higher silhouette score than hierarchical clustering, indicating better cluster cohesion and separation for this dataset.

### Performance Comparison Table
| Clustering Method   | Silhouette Score |
|---------------------|------------------|
| K-Means             | 0.5221           |
| Hierarchical (Ward) | 0.5103           |

## Real-World Applications:
- **Customer Segmentation**: K-Means and hierarchical clustering can group customers by purchasing behavior or demographics, enabling targeted marketing strategies.
- **Image Processing**: Clustering can segment images into regions with similar pixel characteristics, useful in medical imaging or satellite imagery analysis.
- **Anomaly Detection**: Clustering normal data points can help identify outliers (e.g., fraudulent transactions or defective products) by flagging points that don’t belong to any cluster.
- **Genomics**: Clustering gene expression data can reveal patterns associated with diseases or biological processes, guiding medical research.
- **Social Network Analysis**:Clustering identifies communities in social networks by grouping users based on connections. Platforms like LinkedIn use it to suggest connections or detect influential groups, leveraging dendrograms to visualize network structures.
  
## Files:
- `Phase_03.ipynb`: Jupyter notebook containing the complete code, visualizations, and results, including K-Means, hierarchical clustering, PCA, and dendrogram.

## Summary:
- **Model Choice**: K-Means and hierarchical clustering were chosen for their ability to group similar data points without labeled data, suitable for the Iris dataset’s natural grouping into three species.
- **PCA Role**: Reduced 4 features to 2D, facilitating visualization of clusters and reducing computational complexity for clustering algorithms.
- **Clustering Methods**:
  - **K-Means**: Identified 3 clusters using the elbow method, with clear separation in PCA space, though sensitive to initial centroids.
  - **Hierarchical Clustering**: Produced 3 clusters using Ward’s linkage, visualized via a dendrogram and PCA scatter plot, offering insight into the data’s hierarchical structure.
- **Performance**: K-Means slightly outperformed hierarchical clustering based on silhouette scores, reflecting its effectiveness for the Iris dataset’s well-separated clusters.
- **Visualization**: 2D PCA scatter plots and a dendrogram provided clear representations of cluster assignments and hierarchical relationships, confirming the effectiveness of both clustering approaches.
