import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualizations style
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv('Cardiotocographic.csv')

df.head()

## Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

## Handle missing values (example: drop or fill)
df.fillna(df.median(), inplace=True)  

## Check data types
print("Data types:\n", df.dtypes)

## Detect outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define outlier limits
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = ((df < lower_bound) | (df > upper_bound)).sum()
print("Outliers in each column:\n", outliers)

# Statistical Summary
stat_summary = df.describe()
print("Statistical Summary:\n", stat_summary)

## Histograms
df.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()

## Boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Variables")
plt.show()

## Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

## Pair Plots
sns.pairplot(df)
plt.title("Pair Plots of Variables")
plt.show()

## Violin Plots (example for variable 'LB')
plt.figure(figsize=(12, 6))
sns.violinplot(x='LB', data=df)
plt.title("Violin Plot for Baseline Fetal Heart Rate")
plt.show()

## Identify correlations
correlations = df.corr()
print("Correlations:\n", correlations)
