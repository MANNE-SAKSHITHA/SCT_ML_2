import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
customers = pd.read_csv("Mall_Customers.csv")

# Select Features
X = customers[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
customers['Cluster'] = kmeans.fit_predict(X)

# Give Meaningful Names to Clusters
cluster_names = {
    0: "Premium Customers",
    1: "Budget Shoppers",
    2: "Luxury Buyers",
    3: "Regular Customers",
    4: "High Potential"
}

customers['Category'] = customers['Cluster'].map(cluster_names)

# Print Sample Output
print("\nCustomer Categories:")
print(customers[['Annual Income (k$)',
                 'Spending Score (1-100)',
                 'Category']].head())

# Count Customers in Each Cluster
cluster_counts = customers['Category'].value_counts()

# Colors
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Create Figure
plt.figure(figsize=(12,5))

# ----------------------------
# Scatter Plot
# ----------------------------
plt.subplot(1,2,1)

for i, category in cluster_names.items():

    data = customers[customers['Cluster'] == i]

    plt.scatter(
        data['Annual Income (k$)'],
        data['Spending Score (1-100)'],
        color=colors[i],
        s=60,
        label=category
    )

# Plot Centroids
centers = kmeans.cluster_centers_

plt.scatter(
    centers[:,0],
    centers[:,1],
    c='black',
    marker='X',
    s=250,
    label='Centroids'
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8)

# ----------------------------
# Bar Chart
# ----------------------------
plt.subplot(1,2,2)

plt.bar(
    cluster_counts.index,
    cluster_counts.values,
    color=colors
)

plt.title("Customers per Category")
plt.xlabel("Customer Category")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()