#Customer Segmentation using KNN Algos

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D


class KNN:
    def __init__(self, df):
        """
        Initiate the process
        Parameters:
            - self
            - df: DatFrame 
        """
        self.df = df
        self.run_pipeline()

    def run_pipeline(self):
        """
        Running pipeline for automated values
        """
        print("Lets start the flow")
        time.sleep(3)
        print("Running dataset clensing")
        self.df = self.dataset_clean()
    def dataset_clean(self):
        """
        Cleans the datase and indexes minutes and hours to the time invoice daete
        Parameters:
            - self.df: Dataframe from excel
            - index: Indexing minutes and hours
            - fillna(0): fills the nan values with 0 values
            - X, y: Splits for numeric scaling and target values
            - Robust Scaler: Scaler for numeric values
        Returns:
            - self.df: The dataframe
        """
        try:
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            self.df.drop(columns = ['InvoiceNo', 'StockCode', 'Description', 'Country'], inplace = True, errors = 'ignore')
            self.df = self.df.sort_values('InvoiceDate').set_index('InvoiceDate')
            self.df['minute'] = self.df.index.minute #feature engineering
            self.df['hour'] = self.df.index.hour
            self.df = self.df.fillna(0)
            agg = self.df.groupby('CustomerID').agg({ # aggregated sums
                'Quantity': 'sum',
                'UnitPrice': 'sum'
                }).fillna(0)
            agg['t_vals'] = (agg.sum(axis = 1) > 0).astype(int)
            print("Viewing the dataset:\n", self.df.head())
            print("Viewing the info:\n", self.df.info())
            #Creating datasets for splits
            self.target_name = ['Quantity', 'UnitPrice']
            X = self.df.drop(columns = self.target_name)
            y = self.df[self.target_name]
            print(f"Sample Size: X: {X.shape}, Y: {y.shape}")
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            scaler = RobustScaler() # Scaler
            scaled_values = scaler.fit_transform(self.df[num_cols])
            scaled_df = pd.DataFrame(scaled_values, columns = num_cols, index = self.df.index)
            non_numeric_cols = self.df.drop(columns = num_cols) 
            self.df_f = pd.concat([scaled_df, non_numeric_cols], axis = 1)
            self.df = pd.DataFrame(self.df_f, index = self.df_f.index)
            if self.df.index.duplicated().any(): #Error handling for duplicate values
                print("Duplicated index detected. Resetting now")
                self.df = self.df.reset_index(drop = True)
                y = y.reset_index(drop = True)
            self.df[self.target_name] = y.loc[self.df.index]
            print(f"Processed Data\n:", self.df.head())
            print("Are there any missing values?:", self.df.isnull().sum())
            return self.df
        except Exception as e:
            print(f"Error while dataset processing and cleaning{str(e)}")
            raise Exception(e)
    def split(self, test_size):
        """
        Splits the dataset into train and test dataset.
        Parameters: 
            - X, y: 
        Returns:
            -
        """
        try:
            target_cols = ['Quantity', 'UnitPrice']
            X = self.df.drop(columns = target_cols)
            y = self.df[target_cols]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 20)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error during splitting{str(e)}")
            raise Exception(e)


if __name__ == "__main__":
    try:
        df = pd.read_excel("Online Retail.xlsx")
        process = KNN(df)
        print("Splitting the dataset")
        time.sleep(3)
        X_train, X_test, y_train, y_test = process.split(test_size = 0.25)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)
        print("Size of train set: ", X_train.shape, y_train.shape)
        print("Size of test set: ", X_test.shape, y_test.shape)
        print("Viewing the exact number of KNN cluters to be used")
        # For cluster setting of the use for Kmeans and KNeifgborClassification
        best_k = 0
        best_score = -1
        scores = []
        X_samples = X_train.sample(n = 10000, random_state = 42)
        for k in range(2, 11):
            kmeans = KMeans(n_clusters = k, random_state = 42)
            labels = kmeans.fit_predict(X_samples)
            score = silhouette_score(X_samples, labels) #score for the best values
            #silhouette_score measures how well clusters are present in the dataset. Gives as score betwen -1 and 1.
            #1 is point fits very well in the clusters
            #0 is is in the border between 2 clusters
            #-1 means points are in the weong clusters
            scores.append(score)

            if score > best_score:
                best_k = k
                best_score = score


        print("The best cluster value is seelcted as: ", best_k)
        plt.figure(figsize = (8,5))
        plt.plot(range(2, 11), scores, marker = 'o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhoutte Score')
        plt.title('Autoselect Best K')
        plt.grid(True)
        plt.show()
        #Fitting the best value of clusters using PCA
        print("Lets visualize the clusters using the best K")
        time.sleep(3)
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(X_train)
        # Plotting KMeans
        kmeans = KMeans(n_clusters = best_k, random_state = 42)
        cluster_labels = kmeans.fit_predict(X_train)
        plt.figure(figsize = (8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c = cluster_labels, cmap = 'viridis', s = 50)
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], c = "red", s = 200, marker = 'X', label = 'Centers')

        plt.title(f'KMeans Clusters (K = {best_k}) - PCA 2D View')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
        # Visualizing the results in 3D
        print("Now viewing in 3D")
        time.sleep(3)
        pca_3d = PCA(n_components = 3)
        X_train_3d = pca_3d.fit_transform(X_train)

        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], c = cluster_labels, s = 50)
        centers_3d = pca_3d.transform(kmeans.cluster_centers_)
        ax.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2], c = 'red', s = 200, marker = 'X', label = 'Centers' )
        ax.set_title(f'3D Cluster Visualization (K ={best_k})')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        #Now testing the prediction cases and the prediction clusters
        print("Predicting test cases")
        time.sleep(3)
        test_labels =  kmeans.predict(X_test)
        #Kmeans are the clustering algorithms. 
        #Sets the clusters based on the best score from silhouette_score
        X_test_pca = pca.transform(X_test)
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c = test_labels, cmap = 'cool', s = 50)
        plt.title('X_test Cluster Assignment (2D PCA)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.grid(True)
        plt.show()
        #Evaluation of clusters
        print("Evaluating Clusters vs True Labels")
        time.sleep(3)
        ari_1 = adjusted_rand_score(y_train['Quantity'], cluster_labels)
        ari_2 = adjusted_rand_score(y_train['UnitPrice'], cluster_labels)
        #adjusted rand score. To evaluate the quality of a clustering technique.
        #Tells how the clusters are built.
        #Shows the similarity of how the clusters are present.
        print(f"Adjusted Rand Index for Label 'Quantity' (Train): {ari_1: .4f}")
        print(f"Adjusted Rand Index for Label 'UnitPrice' (Train): {ari_2: .4f}")
        nmi_1 = normalized_mutual_info_score(y_train['Quantity'], cluster_labels)
        nmi_2 = normalized_mutual_info_score(y_train['UnitPrice'], cluster_labels)
        #Shares normalized mutual informations between two neighboring clusters
        #0 means no relation
        #1 means perfect correlation
        print(f"Normalized Mutual Information 'Quantity' (Train): {nmi_1:.4f}")
        print(f"Normalized Mutual Information 'UnitPrice' (Train): {nmi_2:.4f}")

        print("Ending the flow. Shutting down in 3 seconds")
        time.sleep(3)
    except Exception as e:
        print(f"Error during model splitting {str(e)}")
        raise Exception(e)
#----------------------------------------------------------This ends the segmentation file---------------------------------------------------------