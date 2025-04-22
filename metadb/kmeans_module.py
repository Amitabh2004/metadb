def kmeans():
    print("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess the dataset
df = pd.read_csv("mall_customers.csv")

# Drop CustomerID if present
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# Normalize relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled_data, columns=features)

# Step 2: Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 3: Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100)
plt.title("Customer Segments (k=3)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

# Step 4: Clusters already assigned in df['Cluster']

# Step 5: Analyze cluster characteristics
print("Cluster Characteristics:")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())""")