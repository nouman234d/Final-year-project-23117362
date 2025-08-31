
**OPTIMISING RETAIL INVENTORY MANAGEMENT USING MACHINE LEARNING AND SALES DATA** 

---

# Overview
This notebook implements a comprehensive demand forecasting framework using the **Online Retail II dataset**. It evaluates multiple forecasting approaches—statistical (SARIMA, Prophet), machine learning (XGBoost), and deep learning (LSTM)—to predict product demand at the Stock Keeping Unit (SKU) level. Additionally, it applies **KMeans clustering** to segment products and compare cluster-level forecasts against SKU-level models, testing whether segmentation enhances predictive accuracy.  

---

# Dataset
- **Path/URL:** Not specified in notebook (loaded as combined Online Retail II CSV/Excel sheets).  
- **Target column:** `y` (daily aggregated demand quantity).  
- **Feature column(s):**  
  - `Quantity`, `UnitPrice`  
  - Engineered: `Sales`, `Month`, `Week`, `Day`, `DayOfWeek`, lag features (1, 7, 30 days), rolling mean, rolling std, holiday indicators  
- **Feature count/types:** Not explicitly printed; includes temporal, numerical, and categorical engineered features.  

---

# Features & Preprocessing
- Removed cancellations (Invoice codes starting with “C”).  
- Filtered out non-positive `Quantity` and `UnitPrice`.  
- Converted datatypes: `InvoiceDate` → datetime; identifiers → string.  
- Created new feature: `Sales = Quantity × UnitPrice`.  
- Aggregated to **daily demand** per SKU (and per cluster for clustering experiments).  
- Engineered features:  
  - **Temporal:** Month, Week, Day, DayOfWeek  
  - **Lag features:** 1-day, 7-day, 30-day lags  
  - **Rolling statistics:** rolling mean, rolling std  
  - **Holiday flags:** UK public holidays (via `holidays` library)  
- For clustering: computed `TotalQuantity`, `TotalSales`, `InvoiceCount`, and `AverageUnitPrice` per SKU, scaled with `StandardScaler`, and segmented with **KMeans (k=3, n_init=10, random_state=42)**.  

---

# Models
- **SARIMA (Seasonal ARIMA):**  
  - Implemented with `statsmodels.SARIMAX`  
  - Tuned seasonal orders `(P,D,Q,m)` with AIC-based search.  
- **Prophet:**  
  - Additive model with yearly & weekly seasonality, holiday effects.  
- **XGBoost (XGBRegressor):**  
  - Parameters: `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `objective='reg:squarederror'`, `random_state=42`.  
- **LSTM (Keras Sequential):**  
  - Architecture: `Input(shape=(seq_len,1))` → `LSTM(64, return_sequences=True)` → `Dropout(0.2)` → `LSTM(32)` → `Dense(1)`.  
  - Loss: `mse`, Optimizer: `adam`, Epochs: 30, Batch size: 32.  

---

# Evaluation
- **Metrics:**  
  - `MAE` (Mean Absolute Error)  
  - `MSE` (Mean Squared Error)  
  - `RMSE` (Root Mean Squared Error)  
- **Visualizations:**  
  - Forecast vs Actual plots (per model, per cluster)  
  - Residual plots  
  - Elbow method (KMeans inertia)  
  - Silhouette score plots (clustering quality)  
- **Tuning:**  
  - SARIMA: grid/loop search over parameters  
  - XGBoost: parameter tuning (learning_rate, max_depth, n_estimators)  
  - LSTM: validation split, early stopping  

---

# Environment & Requirements
- **Libraries:**  
  - pandas, numpy  
  - matplotlib, seaborn  
  - statsmodels (SARIMAX)  
  - prophet  
  - xgboost  
  - scikit-learn (StandardScaler, KMeans, metrics)  
  - holidays  
  - tensorflow / keras  
- **Install example:**  
  ```bash
  pip install pandas numpy matplotlib seaborn statsmodels prophet xgboost scikit-learn holidays tensorflow
  ```
