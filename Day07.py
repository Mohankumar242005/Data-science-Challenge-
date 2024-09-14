import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

file_path = r'C:\Users\mohan\OneDrive\Documents\python\Dataset .csv'
data = pd.read_csv(file_path)
print("Dataset\n",data.head(5))

print("\n Dataset Information \n",df.info())
print("\n Dataset Description \n",df.describe())


print("\n Confidence Interval on Aggregate Rating \n")
ratings = data['Aggregate rating']

mean_rating = np.mean(ratings)
std_err_rating = stats.sem(ratings)

confidence_level = 0.95

confidence_interval = stats.t.interval(confidence_level, len(ratings)-1, loc=mean_rating, scale=std_err_rating)

print(f"Mean Aggregate Rating: {mean_rating}")
print(f"95% Confidence Interval for Aggregate Rating: {confidence_interval}")

print("\n  Correlation Analysis \n")
aggregate_ratings = data['Aggregate rating']
votes = data['Votes']

votes = pd.to_numeric(votes, errors='coerce')

corr_coefficient, _ = pearsonr(aggregate_ratings, votes)

print(f"Pearson Correlation between Aggregate Rating and Votes: {corr_coefficient}")

print("\n Linear Regression \n")
X = votes.values.reshape(-1, 1)
y = aggregate_ratings.values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(votes, aggregate_ratings, color='blue', label='Data Points')
plt.plot(votes, y_pred, color='red', label='Regression Line')
plt.xlabel('Votes')
plt.ylabel('Aggregate Rating')
plt.legend()
plt.title('Linear Regression: Votes vs Aggregate Rating')
plt.show()

print(f"Slope (Coefficient): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")