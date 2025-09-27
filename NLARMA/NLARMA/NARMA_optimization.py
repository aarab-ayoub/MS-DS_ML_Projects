import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv('./simulated_stock_prices.csv')
except FileNotFoundError:
    print("Make sure 'simulated_stock_prices.csv' is in the same directory.")
    exit()

price_series = df['Simulated_Stock_Price'].dropna().to_numpy().reshape(-1, 1)

# --- 2. Feature Engineering: Create Lagged Features ---
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 5
X, y = create_dataset(price_series, look_back)

# --- 3. Split and Scale the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 4. Grid SearchCV for MLPRegressor ---
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01 , 0.05]
}

base_model = MLPRegressor(
    solver='adam',
    max_iter=500,
    random_state=42,
    shuffle=False
)

tscv = TimeSeriesSplit(n_splits=3)

print("\n Starting Grid Search for NARMA MLPRegressor...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train_scaled.ravel())
print("Grid Search Complete.")
print("Best Parameters:", grid_search.best_params_)

# Use the best model found
narma_model = grid_search.best_estimator_
print("\nTraining the best NARMA model...")
narma_model.fit(X_train_scaled, y_train_scaled.ravel())
print("Training complete.")

# --- 5. Make Predictions and Evaluate ---
predictions_scaled = narma_model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"\n NARMA Root Mean Squared Error (RMSE) on Test Data: {rmse:.4f}")

# --- ARIMA Model for Comparison ---
arima_model = ARIMA(y_train, order=(5, 0, 0))
arima_fit = arima_model.fit()
arima_predictions = arima_fit.forecast(steps=len(y_test))
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_predictions))
print(f"ARIMA Model RMSE on Test Data: {arima_rmse:.4f}")

# --- 6. Visualize the Results ---
plt.figure(figsize=(15, 7))
plt.title('NARMA Model: Stock Price Prediction vs. Actual')
plt.ylabel('Simulated Stock Price')
plt.xlabel('Time')
plt.grid(True)

plt.plot(np.arange(len(y_train)), y_train, label='Training Data', color='blue')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual Test Data', color='green')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions, label='Predicted Data', color='red', linestyle='--')
plt.legend()
plt.show()

# --- Visualize ARIMA vs NARMA ---
plt.figure(figsize=(15, 7))
plt.title('ARIMA vs NARMA Predictions')
plt.ylabel('Simulated Stock Price')
plt.xlabel('Time')
plt.grid(True)

plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual Test Data', color='green')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), arima_predictions, label='ARIMA Predictions', color='orange', linestyle='--')
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), predictions, label='NARMA Predictions', color='red', linestyle='--')
plt.legend()
plt.show()


# Calculate residuals for both models
arima_residuals = y_test - arima_predictions
narma_residuals = y_test.flatten() - predictions.flatten() # Ensure shapes match

# Create the time axis for the test set
test_time_axis = np.arange(len(y_train), len(y_train) + len(y_test))

plt.figure(figsize=(15, 7))
plt.title('Residuals (Errors) vs. Time on Test Data')
plt.xlabel('Time')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True)

# Plot residuals for both models
plt.plot(test_time_axis, arima_residuals, label='ARIMA Residuals', color='orange', alpha=0.7)
plt.plot(test_time_axis, narma_residuals, label='Optimized NLARMA Residuals', color='red', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.show()
