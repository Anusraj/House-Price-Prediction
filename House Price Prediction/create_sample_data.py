import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'area': np.random.normal(1500, 500, n_samples),  # Square feet
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'price': np.zeros(n_samples)  # Will be calculated below
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Generate prices based on features
base_price = 200000
df['price'] = (
    base_price +
    (df['area'] * 100) +
    (df['bedrooms'] * 25000) +
    (df['bathrooms'] * 15000) -
    (df['age'] * 1000) +
    np.random.normal(0, 20000, n_samples)
)

# Adjust prices based on location
location_multipliers = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.8}
for location, multiplier in location_multipliers.items():
    df.loc[df['location'] == location, 'price'] *= multiplier

# Save to CSV
df.to_csv('house_data.csv', index=False)
print("Sample dataset created successfully!")