import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

columns = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Locality', 'Longitude', 
           'Latitude', 'Cuisines', 'Price range', 'Aggregate rating', 'Votes']
data = df[columns]

data.fillna(method='ffill', inplace=True)

label_encoder = LabelEncoder()

categorical_cols = ['Restaurant Name', 'City', 'Locality', 'Cuisines']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(['Restaurant ID'], axis=1))

X = data.drop(['Aggregate rating'], axis=1)
y = data['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
tree_preds = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_preds)
print(f"Decision Tree MSE: {tree_mse}")

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)
forest_preds = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_preds)
print(f"Random Forest MSE: {forest_mse}")

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
kmeans_labels = kmeans.labels_

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans_labels)
plt.title('K-Means Clustering')
plt.show()

plt.figure(figsize=(10, 7))
plt.title("Hierarchical Clustering Dendrogram")
dend = shc.dendrogram(shc.linkage(scaled_data, method='ward'))
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_data)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA Result')
plt.show()

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(scaled_data)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('t-SNE Result')
plt.show()
