def hierarchical():
    print("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Step 1: Load dataset and scale features
df = pd.read_csv("Wholesale customers data.csv")
X = df.drop(['Channel', 'Region'], axis=1) if 'Channel' in df.columns and 'Region' in df.columns else df.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply Agglomerative Clustering (Euclidean, Complete)
agglo = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
df['AgglomerativeCluster'] = agglo.fit_predict(X_scaled)

# Step 3: Generate and interpret dendrogram
linked = linkage(X_scaled, method='complete')
plt.figure(figsize=(12, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Dendrogram - Complete Linkage")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 4: Labels are assigned above using 3 clusters

# Step 5: Visualize clusters (using first two PCA features or raw features)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['AgglomerativeCluster'], palette='Set1', s=100)
plt.title("Agglomerative Clustering (2D Projection)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# Step 6: Compare with K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['KMeansCluster'] = kmeans.fit_predict(X_scaled)

# Plot K-Means Clustering result
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['KMeansCluster'], palette='Set2', s=100)
plt.title("K-Means Clustering (2D Projection)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
          """)