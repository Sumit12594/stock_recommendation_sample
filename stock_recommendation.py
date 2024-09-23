import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Sample data (replace with actual historical data)
data = {
    'Stock': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ'],
    'Price': [150.25, 2800.75, 305.50, 3400.00, 330.20, 750.80, 220.40, 160.30, 230.15, 170.50],
    'P/E_Ratio': [28.5, 30.2, 35.7, 58.7, 24.3, 380.5, 75.8, 10.5, 45.6, 18.2],
    'Market_Cap_B': [2500, 1900, 2300, 1700, 950, 750, 550, 480, 500, 450],
    'Dividend_Yield': [0.60, 0, 0.80, 0, 0, 0, 0.12, 2.40, 0.70, 2.50],
    'Beta': [1.2, 1.05, 0.95, 1.3, 1.15, 2.0, 1.7, 1.1, 0.9, 0.5],
    'Revenue_Growth': [0.15, 0.23, 0.18, 0.27, 0.20, 0.50, 0.61, 0.05, 0.10, 0.08],
    'Debt_to_Equity': [1.5, 0.07, 0.5, 1.1, 0.15, 0.3, 0.2, 1.3, 0.7, 0.4],
    'ROE': [0.85, 0.23, 0.40, 0.24, 0.25, 0.20, 0.38, 0.15, 0.35, 0.25],
    'Analyst_Rating': [4.2, 4.5, 4.3, 4.1, 3.9, 3.7, 4.4, 3.8, 4.0, 4.2],
    'Future_Growth': [0.12, 0.15, 0.10, 0.18, 0.08, 0.25, 0.20, 0.05, 0.07, 0.06]  # Target variable
}

df = pd.DataFrame(data)

# Prepare features and target
X = df.drop(['Stock', 'Future_Growth'], axis=1)
y = df['Future_Growth']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Function to predict top 4 stocks
def predict_top_4_stocks(model, data, scaler):
    # Prepare input data
    X_pred = data.drop(['Stock', 'Future_Growth'], axis=1)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    predictions = model.predict(X_pred_scaled)
    
    # Add predictions to dataframe
    data['Predicted_Growth'] = predictions
    
    # Sort by predicted growth and get top 4
    top_4 = data.nlargest(4, 'Predicted_Growth')
    return top_4[['Stock', 'Predicted_Growth']]

# Predict top 4 stocks
top_4_stocks = predict_top_4_stocks(model, df, scaler)

print("Top 4 recommended stocks:")
for i, (index, row) in enumerate(top_4_stocks.iterrows(), 1):
    print(f"{i}. {row['Stock']} (Predicted Growth: {row['Predicted_Growth']:.4f})")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)