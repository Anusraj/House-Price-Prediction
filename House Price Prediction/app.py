import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from house_price_prediction import HousePricePredictor

# Page configuration
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction Analysis")

# Load and train model
@st.cache_data
def load_and_train_model():
    predictor = HousePricePredictor()
    data = predictor.load_data('house_data.csv')
    predictor.preprocess_data()
    predictor.train_model()
    return predictor, data

predictor, df = load_and_train_model()

# Sidebar for prediction inputs
st.sidebar.header("Predict House Price")
area = st.sidebar.number_input("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 3, 2)
location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
age = st.sidebar.slider("Age (years)", 0, 50, 10)

# Make prediction
location_code = {"Urban": 0, "Suburban": 1, "Rural": 2}[location]
features = [area, bedrooms, bathrooms, location_code, age]
prediction = predictor.predict_price(features)

st.sidebar.markdown(f"## Predicted Price\n### ${prediction:,.2f}")

# Main page visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='price', bins=30)
    st.pyplot(fig)

    st.subheader("Price by Location")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='location', y='price')
    st.pyplot(fig)

with col2:
    st.subheader("Price vs Area by Location")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='area', y='price', hue='location')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader("Feature Correlations")
    fig, ax = plt.subplots()
    correlation_data = df[['price', 'area', 'bedrooms', 'bathrooms', 'age']]
    sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig)

# Display dataset statistics
st.subheader("Dataset Statistics")
col3, col4 = st.columns(2)

with col3:
    st.write("Basic Statistics:")
    st.write(df.describe())

with col4:
    st.write("Average Price by Location:")
    st.write(df.groupby('location')['price'].mean())

# After the correlation heatmap in col2
st.subheader("Feature Importance")
importance_df = predictor.feature_importance()
fig, ax = plt.subplots()
importance_df.plot(x='Feature', y=['RF Importance', 'GB Importance'], 
                  kind='bar', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)