import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

true_flow = np.array([130, 123, 24, 57, 46, 27, 72, 100, 49])  # True label for route 970-3001
lstm_predictions = np.array([244, 233, 252, 234, 209, 233, 252, 226, 171])
gru_predictions = np.array([243, 235, 249, 250, 204, 237, 244, 243, 174])
sae_predictions = np.array([156, 171, 170, 164, 183, 187, 169, 107, 144])
sae_fixed_predictions = np.array([207, 147,176, 147, 94, 61, 75, 189, 62])
rnn_predictions = np.array([230, 260, 260, 255, 237, 228, 246, 264, 291])


# Function to calculate and print regression metrics
def evaluate_regression(name, true, predictions):
    mae = mean_absolute_error(true, predictions)
    mse = mean_squared_error(true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predictions)
    mape = np.mean(np.abs((true - predictions) / true)) * 100

    print(f"Regression Metrics for {name}:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared (R²): {r2:.2f}")
    print("\n" + "="*50 + "\n")

# Evaluate each model
evaluate_regression("LSTM", true_flow, lstm_predictions)
evaluate_regression("GRU", true_flow, gru_predictions)
evaluate_regression("SAE", true_flow, sae_predictions)
evaluate_regression("SAE_FIXED", true_flow, sae_fixed_predictions)
evaluate_regression("RNN", true_flow, rnn_predictions)
