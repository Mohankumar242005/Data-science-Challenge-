import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

print("\n Dataset Information \n",df.info())
print("\n Dataset Description \n",df.describe())
print("\n Dataset Types\n",df.dtypes)

print("\n Null values in Dataset \n",df.isnull())
print("\n No Null values in Dataset \n",df.isnull().sum())


df['Longitude']=df['Longitude'].astype('int64')
df['Latitude']=df['Latitude'].astype('int64')
df['Votes']=df['Votes'].astype('float64')

print("\n Updated Dataset \n",df[['Latitude','Longitude','Votes']])

sns.histplot(df['Aggregate rating'], bins=30,kde=True)
plt.title("Distribution of Aggregate ratings")
plt.xlabel("Aggregate rating")
plt.ylabel("Frequency")
plt.show()

class_counts=df['Aggregate rating'].value_counts()
print(class_counts.head())

sns.countplot(x='Aggregate rating',data=df)
plt.title(" class Distribution of Aggregate ratings")
plt.xlabel("Aggregate rating")
plt.ylabel("Count")
plt.show()

mean=df.Votes.mean()
median=df.Votes.median()
mode=df.Votes.mode()
standard_deviation=df.Votes.std()
statistics=df.Votes.describe()

print("\nMean Value :", mean)
print("\nMedian Value :",median)
print(" \nMode Value :",mode)
print(" \nStandard Deviation Value :",standard_deviation)
print(" \nStatistics Values :",statistics)

print("\n Minimum Voute is:",df.Votes.min())
print("\n Maximum Voute is:",df.Votes.max())
print("\n Varience of Voute is:",df.Votes.var())

print("\n Probability Basics \n")

print("\n Independent Events \n")
df1=df[['Restaurant ID','Has Table booking','Is delivering now']]
print(df1.head(5))

P_table_booking = len(df[df['Has Table booking'] == 'Yes']) / len(df)

P_delivering_now = len(df[df['Is delivering now'] == 'Yes']) / len(df)

P_both = P_table_booking * P_delivering_now

print("\n"f"Probability of having table booking: {P_table_booking}")
print("\n"f"Probability of delivering now: {P_delivering_now}")
print("\n"f"Probability of both (independent events): {P_both}")

print("\n Conditional Probability \n")

df2=df[['Restaurant ID','Price range','Rating text']]
print("\n",df2.head(5))

price_range_4 = df[df['Price range'] == 4]
P_excellent_given_price4 = len(price_range_4[price_range_4['Rating text'] == 'Excellent']) / len(price_range_4)

print("\n"f"Probability of Excellent rating given Price range = 4: {P_excellent_given_price4}")

print("\n Bayes' Theorem \n")

df2=df[['Restaurant ID','City','Rating text']]
print("\n",df2.head(5))

P_mandaluyong = len(df[df['City'] == 'Mandaluyong City']) / len(df)

P_excellent = len(df[df['Rating text'] == 'Excellent']) / len(df)

P_excellent_given_mandaluyong = len(df[(df['City'] == 'Mandaluyong City') & (df['Rating text'] == 'Excellent')]) / len(df[df['City'] == 'Mandaluyong City'])

P_mandaluyong_given_excellent = (P_excellent_given_mandaluyong * P_mandaluyong) / P_excellent

print("\n"f"Probability of Mandaluyong given Excellent rating: {P_mandaluyong_given_excellent}")
