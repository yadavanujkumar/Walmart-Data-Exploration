# Walmart Sales Data - Comprehensive Analysis

A deep dive into Walmart sales data using Exploratory Data Analysis (EDA), traditional Machine Learning, and Deep Learning techniques to extract meaningful insights and build accurate predictive models.

## 📊 Project Overview

This project performs comprehensive analysis of Walmart sales data including:
- **Exploratory Data Analysis (EDA)** with statistical insights and visualizations
- **Traditional Machine Learning** models (Linear Regression, Random Forest, Gradient Boosting, etc.)
- **Deep Learning** models (Neural Networks and LSTM for time series prediction)
- **Feature Engineering** for improved model performance
- **Predictive Modeling** for sales forecasting

## 🎯 Key Insights

### Business Insights
- Analyzed 6,435 records across multiple Walmart stores
- Identified seasonal patterns and holiday impacts on sales
- Discovered key economic indicators affecting sales performance
- Quantified the relationship between weather, fuel prices, and consumer behavior

### Model Performance
- Achieved high R² scores (>0.95) with ensemble methods
- Implemented deep learning models with LSTM architecture for time series forecasting
- Comprehensive feature engineering with lag features and rolling statistics
- Model comparison across 8 different algorithms

### Key Findings
- Holiday weeks show significant sales variations
- Strong quarterly seasonality patterns identified
- Temperature and unemployment rate show notable correlations with sales
- Store-level performance varies significantly, enabling targeted strategies

## 📁 Project Structure

```
Walmart-Data-Exploration/
├── walmart.csv                      # Raw dataset
├── walmart_analysis.ipynb          # Main analysis notebook (EDA + ML + DL)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── Generated Files (after running):
    ├── best_traditional_model.pkl  # Saved ML model
    ├── scaler.pkl                  # Feature scaler
    ├── neural_network_model.h5     # Saved NN model
    ├── lstm_model.h5               # Saved LSTM model
    └── Visualizations (PNG files):
        ├── eda_distributions.png
        ├── correlation_matrix.png
        ├── feature_relationships.png
        ├── seasonal_patterns.png
        ├── model_comparison.png
        ├── feature_importance.png
        ├── predictions_vs_actual.png
        ├── nn_training_history.png
        ├── lstm_training_history.png
        ├── final_model_comparison.png
        └── residual_analysis.png
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Walmart-Data-Exploration.git
cd Walmart-Data-Exploration
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook walmart_analysis.ipynb
```

4. Run all cells to execute the complete analysis

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- **Data Loading & Inspection**: Initial data exploration and quality assessment
- **Statistical Summary**: Descriptive statistics and distribution analysis
- **Correlation Analysis**: Relationship between features
- **Temporal Patterns**: Time series analysis, seasonality, and trends
- **Store Performance**: Store-wise sales comparison
- **Visualizations**: 11+ high-quality plots and charts

### 2. Feature Engineering
- **Temporal Features**: Year, month, week, quarter, day of week
- **Lag Features**: Previous 4 weeks of sales data
- **Rolling Statistics**: Moving averages and standard deviations
- **Store Statistics**: Store-level mean and standard deviation
- **Holiday Indicators**: Binary flags for holiday weeks

### 3. Traditional Machine Learning Models
- **Linear Regression**: Baseline model
- **Ridge & Lasso Regression**: Regularized linear models
- **Decision Tree**: Non-linear decision boundaries
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method

### 4. Deep Learning Models
- **Neural Network**: 
  - 4 hidden layers with 256, 128, 64, and 32 neurons
  - Batch normalization and dropout for regularization
  - Adam optimizer with learning rate scheduling
  
- **LSTM Model**:
  - 3 LSTM layers with 128, 64, and 32 units
  - Designed for time series sequence prediction
  - Captures temporal dependencies in sales data

### 5. Model Evaluation & Comparison
- **Metrics**: R² score, RMSE, MAE
- **Cross-validation**: Train/test split with time-based ordering
- **Residual Analysis**: Error distribution and patterns
- **Feature Importance**: Identify key predictors

## 🔍 Key Features

### Data Features
- **Store**: Store identifier
- **Date**: Week date
- **Weekly_Sales**: Sales amount (target variable)
- **Holiday_Flag**: Whether the week includes a holiday
- **Temperature**: Average temperature
- **Fuel_Price**: Regional fuel price
- **CPI**: Consumer Price Index
- **Unemployment**: Unemployment rate

### Engineered Features
- Temporal: Year, Month, Week, Quarter, DayOfWeek
- Lag: Previous 1-4 weeks sales
- Rolling: 4-week moving average and standard deviation
- Store: Store-level mean and standard deviation

## 📊 Results Summary

The analysis provides:
- **Comprehensive EDA** with 11+ visualizations
- **8 different models** trained and evaluated
- **Performance comparison** across all models
- **Saved models** for future predictions
- **Actionable insights** for business decisions

### Best Model Performance
- Model performance varies by use case
- Traditional ML: Random Forest and Gradient Boosting excel
- Deep Learning: Neural Networks achieve competitive results
- LSTM: Best for sequential time series predictions

## 💡 Business Recommendations

Based on the analysis:
1. **Inventory Management**: Adjust stock levels based on predicted sales patterns
2. **Staffing Optimization**: Align workforce with expected demand
3. **Holiday Planning**: Increase resources during holiday weeks
4. **Regional Strategy**: Tailor approaches based on store performance
5. **Economic Monitoring**: Track CPI and unemployment as leading indicators

## 🛠️ Technical Stack

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **Model Persistence**: Joblib, HDF5

## 📝 Usage

### Running the Analysis
```python
# In Jupyter Notebook, run all cells or specific sections:
# 1. Import libraries and load data
# 2. Perform EDA
# 3. Feature engineering
# 4. Train ML models
# 5. Train DL models
# 6. Compare results
```

### Making Predictions
```python
import joblib
import numpy as np

# Load saved model
model = joblib.load('best_traditional_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare features
features = scaler.transform(your_data)

# Make predictions
predictions = model.predict(features)
```

## 📚 Dependencies

See `requirements.txt` for complete list. Key dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- tensorflow >= 2.13.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**dude**

## 🙏 Acknowledgments

- Walmart for the dataset
- Open source community for amazing tools and libraries
- Data science community for best practices and methodologies

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Note**: This is a comprehensive data science project demonstrating end-to-end analysis from EDA to deep learning. The models and insights can be adapted for similar retail sales forecasting problems.