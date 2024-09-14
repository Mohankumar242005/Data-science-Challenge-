import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

print("\n Data Preprocessing - No of missing values \n")
print(df.isnull().sum())

print("\n Selecting features and target variable\n")

X = df[['Longitude', 'Latitude', 'Price range', 'Votes']]
y = df['Aggregate rating']
print("\n Value of X: \n",X)
print("\n Value of y: \n",y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n Values for X-train \n",X_train)
print("\n Values for y-train \n",y_train)
print("\n Values for X_test \n",X_test)
print("\n Values for y_test \n",y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=1.0)  
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
print("\n y_pred_ridge :",y_pred_ridge)

lasso_model = Lasso(alpha=0.1)  
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
print("\n y_pred_lasso :",y_pred_lasso)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)


print("\n"f"Ridge Regression MSE: {ridge_mse}")
print("\n"f"Lasso Regression MSE: {lasso_mse}")



file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

X = df[['Longitude', 'Latitude', 'Price range', 'Votes']]
y = df['Rating text']
print("\n Value of X: \n",X)
print("\n Value of y: \n",y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\n Values for X-train \n",X_train)
print("\n Values for y-train \n",y_train)
print("\n Values for X_test \n",X_test)
print("\n Values for y_test \n",y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # weighted for multiple classes
recall = recall_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

