# PRODIGY_ML_02
## Customer Purchase History Clustering
#Overview
This repository contains a K-Means clustering model designed to group customers of a retail store based on their purchase history. The clustering is based on two features: Annual_Spend and Frequency_of_Purchases. This can help identify different customer segments for targeted marketing strategies.

#Dataset
The dataset used for this clustering model is customer_purchase_history.csv and contains the following columns:

CustomerID: Unique identifier for each customer.
Annual_Spend: Total amount spent by the customer in a year.
Frequency_of_Purchases: Number of purchases made by the customer in a year.
Files
customer_purchase_history.csv: The dataset used for clustering.
kmeans_clustering.py: The Python script implementing the K-Means clustering algorithm.

#Dependencies
Ensure you have the following Python libraries installed:
pandas
numpy
scikit-learn
matplotlib
seaborn

You can install the dependencies using pip:
pip install pandas numpy scikit-learn matplotlib seaborn

#Usage
1. Load the Dataset: Save the dataset to customer_purchase_history.csv.
2.Run the Clustering Script: Execute the Python script to perform K-Means clustering.
python kmeans_clustering.py

3. Script Details:
   import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('customer_purchase_history.csv')

# Features for clustering
X = df[['Annual_Spend', 'Frequency_of_Purchases']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust the number of clusters as needed
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Spend', y='Frequency_of_Purchases', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Clusters')
plt.xlabel('Annual Spend')
plt.ylabel('Frequency of Purchases')
plt.legend(title='Cluster')
plt.show()

# Print cluster centers
print("Cluster Centers:")
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: Annual Spend: {center[0]:.2f}, Frequency of Purchases: {center[1]:.2f}")

4. Interpreting Results:
The scatter plot will show the different customer clusters.
The printed cluster centers provide insights into the average Annual_Spend and Frequency_of_Purchases for each cluster.

## SCREENSHOTS OF OUTPUT 
![Screenshot 2024-09-07 185851](https://github.com/user-attachments/assets/a1d36c87-ad82-4dd7-8d66-0f1d65d927d5)
![Screenshot 2024-09-07 185906](https://github.com/user-attachments/assets/6f6cf40f-144f-4b2c-93c7-1d5633cd7f6c)

# Notes
You may adjust the number of clusters (n_clusters) based on your specific needs.
Ensure that customer_purchase_history.csv is in the same directory as the script or provide the full path to the file.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

