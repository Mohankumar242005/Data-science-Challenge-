import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

print(df.info())

label_encoder = LabelEncoder()
df['Has Table booking'] = label_encoder.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = label_encoder.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = label_encoder.fit_transform(df['Is delivering now'])

df['City'] = label_encoder.fit_transform(df['City'])
df['Cuisines'] = label_encoder.fit_transform(df['Cuisines'])

df_cleaned = df.drop(['Restaurant ID', 'Restaurant Name','Locality Verbose', 'Address', 'Locality', 'Currency', 'Switch to order menu','Rating color', 'Rating text'], axis=1)

print(df_cleaned.head())

X = df_cleaned.drop('Aggregate rating', axis=1)
y = df_cleaned['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

model = RandomForestRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance in Random Forest Model")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Votes'], y=df['Aggregate rating'])
plt.title('Votes vs Aggregate Rating')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Price range'], bins=4, kde=True)
plt.title('Distribution of Price Range')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()