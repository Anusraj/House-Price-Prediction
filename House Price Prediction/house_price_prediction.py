import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class HousePricePredictor:
    def __init__(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        return self.data
    
    def preprocess_data(self):
        features = ['area', 'bedrooms', 'bathrooms', 'location', 'age']
        self.data['location'] = pd.Categorical(self.data['location']).codes
        
        X = self.data[features]
        y = self.data['price']
        
        # Create polynomial features
        X_poly = self.poly.fit_transform(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_model(self):
        # Train both models
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        self.gb_model.fit(self.X_train_scaled, self.y_train)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        gb_pred = self.gb_model.predict(self.X_test_scaled)
        
        # Ensemble prediction (average of both models)
        self.ensemble_pred = (rf_pred + gb_pred) / 2
        
    def evaluate_model(self):
        # Cross-validation scores
        rf_cv_scores = cross_val_score(self.rf_model, self.X_train_scaled, 
                                     self.y_train, cv=5)
        gb_cv_scores = cross_val_score(self.gb_model, self.X_train_scaled, 
                                     self.y_train, cv=5)
        
        # Calculate metrics for ensemble
        mse = mean_squared_error(self.y_test, self.ensemble_pred)
        r2 = r2_score(self.y_test, self.ensemble_pred)
        
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2 Score': r2,
            'RF CV Score': rf_cv_scores.mean(),
            'GB CV Score': gb_cv_scores.mean()
        }
    
    def predict_price(self, features):
        features_poly = self.poly.transform([features])
        features_scaled = self.scaler.transform(features_poly)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(features_scaled)
        gb_pred = self.gb_model.predict(features_scaled)
        
        # Return ensemble prediction
        return (rf_pred[0] + gb_pred[0]) / 2
    
    def feature_importance(self):
        feature_names = ['area', 'bedrooms', 'bathrooms', 'location', 'age']
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        return pd.DataFrame({
            'Feature': feature_names,
            'RF Importance': rf_importance[:len(feature_names)],
            'GB Importance': gb_importance[:len(feature_names)]
        })