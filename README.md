# ClusterAnalysis
## Customer-Segmentation
In this project, I developed an end-to-end customer segmentation pipeline using unsupervised machine learning to uncover distinct customer groups from retail transaction data. The process began with comprehensive data preprocessing: irrelevant columns were removed, missing values handled, and time-based features (hour, minute) engineered from invoice timestamps. Aggregated metrics such as total quantity and unit price per customer were computed to derive meaningful behavioral insights.

I applied RobustScaler to normalize numerical features and ensure model robustness against outliers. To determine the optimal number of clusters, I used the silhouette score across a range of K values (2–10), selecting the value that maximized clustering effectiveness. Dimensionality reduction via PCA allowed for intuitive 2D and 3D visualization of clusters and centroids, offering a clear view into customer segmentation structure.

The final model leveraged the K-Means algorithm for clustering, followed by evaluation using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) to compare clustering results with actual behavioral metrics. While the evaluation involved continuous targets, it still provided insights into clustering performance. This project highlights my ability to design scalable machine learning workflows, perform insightful data transformations, and communicate results through visualizations.

## Images
### AutoSelect Best_K
![AutoSelect_Best_K](https://github.com/user-attachments/assets/ecdd6cf4-6238-4df9-8ae0-2dd38d2b85b5)

### KMeans Cliustering PCA 2D
![Kmeans_clustering_pca2d](https://github.com/user-attachments/assets/44dca3c6-15a6-44db-85e2-46e8ba5de20d)

### Test Cluster PCA 3D
![Test_Cluster_PCA3D](https://github.com/user-attachments/assets/ceeae77a-bfeb-4d60-a8c7-29cb2f54f917)

### Adjusted Rand Score
![Adjusted_Rand_Score](https://github.com/user-attachments/assets/3efec6a4-5c7b-426f-9e70-d7f0e98941e1)

### Normalized Mutual Info
![Normalized_Mutual_Info](https://github.com/user-attachments/assets/b8d4c4c2-6617-4c7f-8bb7-2d732a3c3a47)

