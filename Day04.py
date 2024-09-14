import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df=pd.read_csv("iris.csv")
print(df.head(5))
print("\n INFORMATION ABOUT DATASET \n")
print(df.info())
print("\n DESCRIPTION ABOUT THE DATA \n")
print(df.describe())
print("\n DATA TYPES OF EACH COLUMN \n")
print(df.dtypes)

print("\n CLEANING THE DATA -Handling Missing Values\n\n")
print(df.notna())
print("\n CLEANING THE DATA-isnull \n")
print(df.isnull())
print("\n CLEANING THE DATA-isnull.sum \n")
print(df.isnull().sum())
print("\n CLEANING THE DATA-dropna \n")
print(df.dropna().head(5))
print("\n CLEANING THE DATA-fillna \n")
print(df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean(), inplace=True))


print("\n CLEANING THE DATA - Detecting and handling Outliers \n\n")

print("\n Boxplot \n")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['SepalLengthCm'])
plt.show()

print("\n Removal of outliers \n")
Q1 = df['SepalLengthCm'].quantile(0.25)
Q3 = df['SepalLengthCm'].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df['SepalLengthCm'] < (Q1 - 1.5 * IQR)) | (df['SepalLengthCm'] > (Q3 + 1.5 * IQR)))]

print(df.head(10))

print("\n\n Exploratory analysis \n \n")

print("\n\n Univariate Analysis \n\n")

df['SepalWidthCm'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Numerical Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Species', y='SepalWidthCm', data=df)
plt.title('Boxplot of Categorical vs Numerical Column')
plt.show()

print("\n\n Bivariate Analysis \n\n")

sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=df)
plt.title('Scatter Plot of SepalLengthCm vs PetalLengthCm')
plt.show()

df = df.select_dtypes(include=[np.number])

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

print("\n\n Multivariate Analysis \n\n")

sns.pairplot(df)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title('Heatmap of Feature Correlations')
plt.show()

print("\n\n Feature Engineering \n\n")

df['Sepal'] = df['SepalLengthCm'] + df['SepalWidthCm']
print(df.head(5))

df = df.drop(['SepalLengthCm', 'SepalWidthCm'], axis=1)
print(df.head(5))
