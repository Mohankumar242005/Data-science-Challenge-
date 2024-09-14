import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

label_encoder = LabelEncoder()

df['City'] = label_encoder.fit_transform(df['City'])
df['Cuisines'] = label_encoder.fit_transform(df['Cuisines'])
df['Rating text'] = label_encoder.fit_transform(df['Rating text'])


scaler = StandardScaler()
df[['Longitude', 'Latitude', 'Aggregate rating']] = scaler.fit_transform(df[['Longitude', 'Latitude', 'Aggregate rating']])


X = df[['City', 'Cuisines', 'Longitude', 'Latitude', 'Price range', 'Aggregate rating']]
y = df['Rating text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df= df.drop(columns=['Restaurant Name','Address','Locality','Locality Verbose','Currency','Has Table booking','Has Online delivery','Is delivering now','Switch to order menu','Rating color'])

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("\n Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("\n Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print("\n Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\n Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)
print("\n KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\n KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


