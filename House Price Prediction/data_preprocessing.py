import pandas as pd
import numpy as np

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna({
        'area': df['area'].mean(),
        'bedrooms': df['bedrooms'].median(),
        'bathrooms': df['bathrooms'].median(),
        'age': df['age'].mean()
    })
    
    # Remove outliers using IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5*IQR) & 
               (df[column] <= Q3 + 1.5*IQR)]
        return df
    
    numerical_columns = ['area', 'price', 'age']
    for col in numerical_columns:
        df = remove_outliers(df, col)
    
    return df