# Project Architecture and Workflow

## 📁 Project Structure

```
Walmart-Data-Exploration/
│
├── 📊 DATA
│   └── walmart.csv                    # Raw sales data (6,435 records)
│
├── 📓 ANALYSIS NOTEBOOKS
│   └── walmart_analysis.ipynb         # Complete EDA + ML + DL analysis
│
├── 🐍 PYTHON SCRIPTS
│   ├── train_model.py                 # Quick model training script
│   └── predict.py                     # Prediction script
│
├── 📚 DOCUMENTATION
│   ├── README.md                      # Main project documentation
│   ├── QUICKSTART.md                  # 5-minute getting started guide
│   ├── INSIGHTS.md                    # Detailed analysis insights
│   ├── CONTRIBUTING.md                # Contribution guidelines
│   └── ARCHITECTURE.md                # This file
│
├── ⚙️ CONFIGURATION
│   ├── requirements.txt               # Python dependencies
│   └── .gitignore                     # Git ignore rules
│
├── 📜 LICENSE
│   └── LICENSE                        # MIT License
│
└── 🎯 GENERATED OUTPUTS (after running)
    ├── Models/
    │   ├── best_model.pkl             # Best traditional ML model
    │   ├── best_traditional_model.pkl # Alternative saved model
    │   ├── neural_network_model.h5    # Deep learning NN model
    │   ├── lstm_model.h5              # LSTM time series model
    │   ├── feature_scaler.pkl         # StandardScaler object
    │   └── scaler.pkl                 # Alternative scaler
    │
    ├── Feature Info/
    │   └── feature_names.txt          # List of features used
    │
    └── Visualizations/
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

---

## 🔄 Data Flow Architecture

```
┌─────────────────┐
│  walmart.csv    │  Raw Data
│  (6,435 rows)   │  8 features
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   DATA LOADING & VALIDATION         │
│   - Load CSV with pandas            │
│   - Check for missing values        │
│   - Validate data types             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   FEATURE ENGINEERING               │
│   ├─ Temporal Features (6)          │
│   │  └─ Year, Month, Week, etc.     │
│   ├─ Lag Features (4)               │
│   │  └─ Previous 1-4 weeks sales    │
│   ├─ Rolling Stats (2)              │
│   │  └─ Mean & Std over 4 weeks     │
│   └─ Store Statistics (2)           │
│      └─ Store mean & std            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   DATA PREPARATION                  │
│   - Remove NaN (from lag features)  │
│   - Sort chronologically            │
│   - Split train/test (80/20)       │
│   - Scale features (StandardScaler) │
└────────┬────────────────────────────┘
         │
         ├──────────┬──────────┬───────────┐
         ▼          ▼          ▼           ▼
    ┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐
    │  ML    │ │  ML    │ │   DL    │ │   DL    │
    │ Linear │ │ Ensemble│ │ Neural  │ │  LSTM   │
    │ Models │ │ Models │ │ Network │ │  Model  │
    └───┬────┘ └───┬────┘ └────┬────┘ └────┬────┘
        │          │           │           │
        ▼          ▼           ▼           ▼
    ┌──────────────────────────────────────────┐
    │         MODEL EVALUATION                  │
    │  - R² Score, RMSE, MAE                   │
    │  - Feature Importance                     │
    │  - Residual Analysis                      │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │     MODEL SELECTION & SAVING              │
    │  - Compare all models                     │
    │  - Select best performer                  │
    │  - Save models (.pkl, .h5)               │
    │  - Generate visualizations                │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │          PREDICTION                       │
    │  - Load saved model                       │
    │  - Prepare input features                 │
    │  - Scale features                         │
    │  - Make prediction                        │
    │  - Return sales forecast                  │
    └───────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing Pipeline
```
Raw Data → Date Parsing → Feature Extraction → Lag Creation → 
Rolling Stats → Store Aggregation → NaN Removal → Train/Test Split → 
Feature Scaling → Ready for Modeling
```

### 2. Model Training Pipeline
```
Scaled Features → Model Initialization → Hyperparameter Setting → 
Model Training → Prediction on Train Set → Prediction on Test Set → 
Metric Calculation → Model Comparison → Best Model Selection → 
Model Persistence
```

### 3. Prediction Pipeline
```
New Data → Feature Engineering → Feature Alignment → 
Feature Scaling → Model Loading → Prediction → Output
```

---

## 🎯 Model Architecture Details

### Traditional ML Models

#### Random Forest Architecture
```
Input Features (19) 
    → Bootstrap Sampling
    → Decision Tree 1
    → Decision Tree 2
    → ...
    → Decision Tree 100
    → Majority Vote
    → Predicted Sales
```

#### Gradient Boosting Architecture
```
Input Features (19)
    → Weak Learner 1 (depth=5)
    → Residual Calculation
    → Weak Learner 2 (learns from residuals)
    → ...
    → Weak Learner 100
    → Weighted Sum
    → Predicted Sales
```

### Deep Learning Models

#### Neural Network Architecture
```
Input Layer (19 features)
    ↓
Dense(256) → ReLU → BatchNorm → Dropout(0.3)
    ↓
Dense(128) → ReLU → BatchNorm → Dropout(0.3)
    ↓
Dense(64) → ReLU → BatchNorm → Dropout(0.2)
    ↓
Dense(32) → ReLU → Dropout(0.2)
    ↓
Output Layer (1) → Linear
    ↓
Predicted Sales
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Epochs: Up to 100 (early stopping)
- Batch Size: 64

#### LSTM Architecture
```
Input Sequences (10 timesteps × 19 features)
    ↓
LSTM(128) → Tanh → Dropout(0.3) → Return Sequences
    ↓
LSTM(64) → Tanh → Dropout(0.3) → Return Sequences
    ↓
LSTM(32) → Tanh → Dropout(0.2)
    ↓
Dense(16) → ReLU
    ↓
Output Layer (1) → Linear
    ↓
Predicted Sales
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Sequence Length: 10 weeks
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Epochs: Up to 50
- Batch Size: 64

---

## 📊 Feature Engineering Flow

```
Original Features (8):
├─ Store
├─ Date              ─────┐
├─ Weekly_Sales      ──┐  │
├─ Holiday_Flag        │  │
├─ Temperature         │  │
├─ Fuel_Price          │  │
├─ CPI                 │  │
└─ Unemployment        │  │
                       │  │
     ┌─────────────────┘  │
     │                    │
     ▼                    ▼
Lag Features (4):    Temporal Features (6):
├─ Sales_Lag_1       ├─ Year
├─ Sales_Lag_2       ├─ Month
├─ Sales_Lag_3       ├─ Week
└─ Sales_Lag_4       ├─ Quarter
     │               ├─ Day
     ▼               └─ DayOfWeek
Rolling Stats (2):        │
├─ Sales_Rolling_Mean_4   │
└─ Sales_Rolling_Std_4    │
     │                    │
     └────────┬───────────┘
              ▼
      Store Statistics (2):
      ├─ Store_Mean_Sales
      └─ Store_Std_Sales
              │
              ▼
    Final Feature Set (19)
```

---

## 🔄 Training vs Prediction Workflow

### Training Workflow
```
1. Load Historical Data (walmart.csv)
2. Perform Feature Engineering
3. Split into Train/Test
4. Scale Features
5. Train Multiple Models
6. Evaluate & Compare
7. Select Best Model
8. Save Model + Scaler
9. Generate Reports
```

### Prediction Workflow
```
1. Load Saved Model + Scaler
2. Receive New Input Data
3. Apply Same Feature Engineering
4. Scale with Saved Scaler
5. Make Prediction
6. Return Forecast
```

---

## 🎛️ Component Interactions

```
┌──────────────────────────────────────────────────┐
│              User Interface                       │
│  (Jupyter Notebook / CLI Scripts)                │
└─────┬────────────────────────────────────────────┘
      │
      ├─── Data Layer ─────────────────────────────┐
      │    ├─ walmart.csv (storage)                │
      │    └─ pandas DataFrame (in-memory)         │
      │                                             │
      ├─── Processing Layer ───────────────────────┤
      │    ├─ Feature Engineering Module           │
      │    ├─ Data Validation Module               │
      │    └─ Scaling Module (StandardScaler)      │
      │                                             │
      ├─── Model Layer ────────────────────────────┤
      │    ├─ Traditional ML (scikit-learn)        │
      │    │  ├─ Linear Models                     │
      │    │  └─ Ensemble Models                   │
      │    └─ Deep Learning (TensorFlow/Keras)     │
      │       ├─ Neural Network                    │
      │       └─ LSTM                              │
      │                                             │
      ├─── Evaluation Layer ───────────────────────┤
      │    ├─ Metrics Calculation                  │
      │    ├─ Model Comparison                     │
      │    └─ Visualization Generation             │
      │                                             │
      └─── Persistence Layer ──────────────────────┤
           ├─ Model Serialization (.pkl, .h5)      │
           ├─ Scaler Serialization                 │
           └─ Results Export (images, reports)     │
```

---

## 🚀 Execution Paths

### Path 1: Complete Analysis (Jupyter Notebook)
```
Start → Launch Notebook → Run All Cells →
EDA → Feature Engineering → ML Training →
DL Training → Evaluation → Visualization →
Insights Generation → Model Saving → End
```
**Time**: 10-20 minutes  
**Output**: All visualizations, models, comprehensive insights

### Path 2: Quick Training (train_model.py)
```
Start → Load Data → Engineer Features →
Train Best Models → Evaluate → Save Best Model → End
```
**Time**: 2-5 minutes  
**Output**: best_model.pkl, metrics summary

### Path 3: Prediction (predict.py)
```
Start → Load Model → Get Input → Engineer Features →
Scale → Predict → Display Result → End
```
**Time**: <1 second  
**Output**: Sales prediction

---

## 📈 Performance Optimization

### Data Processing
- **Vectorization**: Use pandas/numpy operations instead of loops
- **Memory**: Use appropriate data types (int32 vs int64)
- **Chunking**: Process large datasets in chunks if needed

### Model Training
- **Parallelization**: Use n_jobs=-1 for tree-based models
- **Early Stopping**: Stop training when validation loss plateaus
- **Batch Size**: Optimize for GPU memory (64 is good default)

### Inference
- **Model Loading**: Load model once, reuse for multiple predictions
- **Batch Prediction**: Predict multiple samples at once
- **Feature Caching**: Cache store statistics for fast lookup

---

## 🔐 Error Handling Flow

```
User Input
    ↓
Input Validation
    ├─ Valid → Continue
    └─ Invalid → Error Message → Retry
        ↓
Feature Engineering
    ├─ Success → Continue
    └─ Failure → Log Error → Graceful Degradation
        ↓
Model Loading
    ├─ Found → Continue
    └─ Not Found → Error: Run training first
        ↓
Prediction
    ├─ Success → Return Result
    └─ Failure → Log Error → Return Error Message
```

---

## 🧪 Testing Architecture

### Unit Tests (Recommended)
- Test data loading
- Test feature engineering functions
- Test model initialization
- Test prediction pipeline

### Integration Tests (Recommended)
- Test end-to-end training
- Test end-to-end prediction
- Test model saving/loading

### Performance Tests (Recommended)
- Benchmark training time
- Benchmark prediction latency
- Monitor memory usage

---

## 📦 Deployment Architecture (Future)

### Production Ready Components
```
┌────────────────────────────────────────┐
│         Web API (Flask/FastAPI)        │
│  ├─ /predict endpoint                  │
│  ├─ /batch-predict endpoint            │
│  └─ /model-info endpoint               │
└──────────────┬─────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    ┌────────┐   ┌─────────┐
    │ Model  │   │ Feature │
    │ Server │   │ Store   │
    └────────┘   └─────────┘
        │             │
        └──────┬──────┘
               ▼
    ┌──────────────────┐
    │  Monitoring &    │
    │  Logging         │
    └──────────────────┘
```

---

## 🎓 Learning Path Through Codebase

**Beginner**: 
1. Start with QUICKSTART.md
2. Read README.md
3. Explore walmart.csv
4. Run predict.py with example

**Intermediate**:
1. Run train_model.py
2. Explore walmart_analysis.ipynb (EDA sections)
3. Understand feature engineering
4. Read INSIGHTS.md

**Advanced**:
1. Deep dive into notebook (ML/DL sections)
2. Experiment with hyperparameters
3. Add new models
4. Optimize performance
5. Deploy to production

---

**Last Updated**: 2025  
**Version**: 1.0
