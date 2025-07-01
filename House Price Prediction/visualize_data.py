import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('house_data.csv')

# Set up the plotting style
plt.style.use('seaborn')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Price Distribution
sns.histplot(data=df, x='price', bins=30, ax=axes[0,0])
axes[0,0].set_title('House Price Distribution')
axes[0,0].set_xlabel('Price ($)')

# 2. Price vs Area Scatter Plot
sns.scatterplot(data=df, x='area', y='price', hue='location', ax=axes[0,1])
axes[0,1].set_title('Price vs Area by Location')
axes[0,1].set_xlabel('Area (sq ft)')
axes[0,1].set_ylabel('Price ($)')

# 3. Average Price by Location
sns.boxplot(data=df, x='location', y='price', ax=axes[1,0])
axes[1,0].set_title('Price Distribution by Location')
axes[1,0].set_xlabel('Location')
axes[1,0].set_ylabel('Price ($)')

# 4. Correlation between Features
correlation_data = df[['price', 'area', 'bedrooms', 'bathrooms', 'age']]
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Feature Correlations')

plt.tight_layout()
plt.savefig('house_price_analysis.png')
plt.show()

# Print basic statistics
print("\nBasic Statistics:")
print(df.describe())

print("\nAverage Price by Location:")
print(df.groupby('location')['price'].mean())