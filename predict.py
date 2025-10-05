#!/usr/bin/env python3
"""
Walmart Sales Prediction Script
Make predictions using trained models
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_model():
    """Load trained model and scaler"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Load feature names
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        return model, scaler, feature_names
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first.")
        return None, None, None


def prepare_prediction_data(data_dict, feature_names):
    """Prepare input data for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found. Setting to 0.")
            df[feature] = 0
    
    # Select features in correct order
    X = df[feature_names]
    
    return X


def predict_sales(input_data):
    """Make sales prediction"""
    # Load model
    model, scaler, feature_names = load_model()
    
    if model is None:
        return None
    
    # Prepare data
    X = prepare_prediction_data(input_data, feature_names)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction


def example_prediction():
    """Example prediction with sample data"""
    print("=" * 70)
    print("WALMART SALES PREDICTION")
    print("=" * 70)
    
    # Example input data
    sample_data = {
        'Store': 1,
        'Holiday_Flag': 0,
        'Temperature': 60.0,
        'Fuel_Price': 2.75,
        'CPI': 211.5,
        'Unemployment': 8.0,
        'Year': 2010,
        'Month': 6,
        'Week': 24,
        'Quarter': 2,
        'DayOfWeek': 4,
        'Sales_Lag_1': 1500000,
        'Sales_Lag_2': 1480000,
        'Sales_Lag_3': 1520000,
        'Sales_Lag_4': 1510000,
        'Sales_Rolling_Mean_4': 1502500,
        'Sales_Rolling_Std_4': 16583,
        'Store_Mean_Sales': 1550000,
        'Store_Std_Sales': 200000
    }
    
    print("\nInput Features:")
    print("-" * 70)
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 70)
    print("Making prediction...")
    
    prediction = predict_sales(sample_data)
    
    if prediction is not None:
        print(f"\nPredicted Weekly Sales: ${prediction:,.2f}")
        print("=" * 70)
    else:
        print("\nPrediction failed. Please check model files.")
        print("=" * 70)


def interactive_prediction():
    """Interactive prediction mode"""
    print("\n" + "=" * 70)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 70)
    
    try:
        # Get user input
        print("\nEnter feature values (press Enter to use default):")
        
        data = {}
        defaults = {
            'Store': 1,
            'Holiday_Flag': 0,
            'Temperature': 60.0,
            'Fuel_Price': 2.75,
            'CPI': 211.5,
            'Unemployment': 8.0,
            'Year': 2010,
            'Month': 6,
            'Week': 24,
            'Quarter': 2,
            'DayOfWeek': 4,
            'Sales_Lag_1': 1500000,
            'Sales_Lag_2': 1480000,
            'Sales_Lag_3': 1520000,
            'Sales_Lag_4': 1510000,
            'Sales_Rolling_Mean_4': 1502500,
            'Sales_Rolling_Std_4': 16583,
            'Store_Mean_Sales': 1550000,
            'Store_Std_Sales': 200000
        }
        
        for key, default_value in defaults.items():
            user_input = input(f"{key} (default: {default_value}): ").strip()
            if user_input:
                try:
                    data[key] = float(user_input) if '.' in str(default_value) or 'Lag' in key or 'Rolling' in key or 'Mean' in key or 'Std' in key else int(user_input)
                except ValueError:
                    print(f"  Invalid input, using default: {default_value}")
                    data[key] = default_value
            else:
                data[key] = default_value
        
        # Make prediction
        prediction = predict_sales(data)
        
        if prediction is not None:
            print("\n" + "=" * 70)
            print(f"PREDICTED WEEKLY SALES: ${prediction:,.2f}")
            print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_prediction()
    else:
        example_prediction()
        print("\nTip: Run with --interactive flag for custom predictions")
        print("Example: python predict.py --interactive")
