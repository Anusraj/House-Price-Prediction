from house_price_prediction import HousePricePredictor
from data_preprocessing import clean_data
import pandas as pd

def main():
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load and preprocess data
    data = predictor.load_data('house_data.csv')
    data = clean_data(data)
    
    # Train model
    predictor.preprocess_data()
    predictor.train_model()
    
    # Evaluate model
    metrics = predictor.evaluate_model()
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Generate visualization
    predictor.plot_predictions()
    
    # Example prediction
    sample_house = [1500, 3, 2, 1, 5]  # area, bedrooms, bathrooms, location, age
    predicted_price = predictor.predict_price(sample_house)
    print(f"\nPredicted price for sample house: ${predicted_price:,.2f}")

if __name__ == "__main__":
    main()