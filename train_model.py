#!/usr/bin/env python3
"""
Walmart Sales Prediction - Quick Training Script
This script provides a streamlined way to train and evaluate models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath='walmart.csv'):
    """Load and prepare Walmart sales data"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Convert date and extract features
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Sort by date
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    
    # Create lag features
    for lag in [1, 2, 3, 4]:
        df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)
    
    # Rolling statistics
    df['Sales_Rolling_Mean_4'] = df.groupby('Store')['Weekly_Sales'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )
    df['Sales_Rolling_Std_4'] = df.groupby('Store')['Weekly_Sales'].transform(
        lambda x: x.rolling(4, min_periods=1).std()
    )
    
    # Store statistics
    store_stats = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
    store_stats.columns = ['Store', 'Store_Mean_Sales', 'Store_Std_Sales']
    df = df.merge(store_stats, on='Store', how='left')
    
    # Remove NaN values
    df = df.dropna()
    
    print(f"Data loaded: {len(df)} records")
    return df


def prepare_features(df):
    """Prepare features for modeling"""
    feature_cols = [
        'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Year', 'Month', 'Week', 'Quarter', 'DayOfWeek',
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_4',
        'Sales_Rolling_Mean_4', 'Sales_Rolling_Std_4',
        'Store_Mean_Sales', 'Store_Std_Sales'
    ]
    
    X = df[feature_cols]
    y = df['Weekly_Sales']
    
    return X, y, feature_cols


def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: ${test_rmse:,.2f}")
        print(f"  Test MAE: ${test_mae:,.2f}")
    
    return results


def main():
    """Main execution function"""
    print("=" * 70)
    print("WALMART SALES PREDICTION - QUICK TRAINING")
    print("=" * 70)
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, feature_cols = prepare_features(df)
    
    # Split data chronologically
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"Test RMSE: ${results[best_model_name]['test_rmse']:,.2f}")
    print("=" * 70)
    
    # Save best model and scaler
    print("\nSaving models...")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    # Save feature names for later use
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print("Models saved:")
    print("  - best_model.pkl")
    print("  - feature_scaler.pkl")
    print("  - feature_names.txt")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\nTop 10 Most Important Features:")
        importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, row in importances.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
