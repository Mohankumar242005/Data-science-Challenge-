import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind


file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
df = pd.read_csv(file_path)
print("Dataset\n",df.head(5))

print("\n Data Preprocessing\n")

print(df.isnull().sum())

df1=df[['Aggregate rating', 'Price range', 'Votes']]
print(df1.head(5))

df['Price range'] = pd.to_numeric(df['Price range'])
df['Aggregate rating'] = pd.to_numeric(df['Aggregate rating'])
df['Votes'] = pd.to_numeric(df['Votes'])

print(df1.head(5))

print("\nAnalyzing Probability Distributions\n")

print("\nNormal Distribution \n")

sns.histplot(df['Aggregate rating'], kde=True)
plt.title('Aggregate Rating Distribution')
plt.show()

sns.histplot(df['Price range'], kde=True)
plt.title('Price range Distribution')
plt.show()

sns.histplot(df['Votes'], kde=True)
plt.title('Votes Distribution')
plt.show()

print("\nBinomial Distribution\n")

success = (df['Aggregate rating'] > 4).sum()
total = len(df)

binomial_dist = binom.pmf(k=range(0, total), n=total, p=success/total)
plt.bar(range(0, total), binomial_dist)
plt.title('Binomial Distribution: Success in Ratings > 4')
plt.show()

success = (df['Price range'] > 2).sum()
total = len(df)

binomial_dist = binom.pmf(k=range(0, total), n=total, p=success/total)
plt.bar(range(0, total), binomial_dist)
plt.title('Binomial Distribution: Success in Price range > 2')
plt.show()

success = (df['Votes'] > 200).sum()
total = len(df)

binomial_dist = binom.pmf(k=range(0, total), n=total, p=success/total)
plt.bar(range(0, total), binomial_dist)
plt.title('Binomial Distribution: Success in Votes > 200')
plt.show()

print("\n Poisson Distribution\n")

mean_votes = df['Votes'].mean()
poisson_dist = poisson.pmf(k=range(0, 1000), mu=mean_votes)  
plt.plot(range(0, 1000), poisson_dist)
plt.title('Poisson Distribution: Number of Votes')
plt.show()

print("\nHypothesis Testing\n")
print("\nOne-Sample t-test\n")

t_stat, p_value = ttest_1samp(df['Aggregate rating'], 4)
print(f"One-sample t-test p-value: {p_value}")

print("\nTwo-Sample t-test \n")

group1 = df[df['Price range'] == 3]['Aggregate rating']
group2 = df[df['Price range'] == 4]['Aggregate rating']

t_stat2, p_value2 = ttest_ind(group1, group2)
print(f"Two-sample t-test p-value: {p_value2}")

print("\nVisualizing Relationships\n")

print("\nVotes vs Rating\n")

sns.scatterplot(x='Votes', y='Aggregate rating', data=df)
plt.title('Votes vs Aggregate Rating')
plt.show()

print("\nRating vs Price Range\n")

sns.boxplot(x='Price range', y='Aggregate rating', data=df)
plt.title('Rating vs Price Range')
plt.show()