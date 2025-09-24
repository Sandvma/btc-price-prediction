import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    """
    Load the BTC‑USD dataset from a CSV file and sort by date.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the historical price data.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dates and sorted chronologically.
    """
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_features(df: pd.DataFrame, n_lags: int = 10):
    """
    Construct lagged return features for time‑series prediction.

    The function computes the daily percentage change of the adjusted
    closing price and then creates lagged versions of this return to
    be used as features for a regression model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least an 'Adj Close' column.
    n_lags : int, optional
        Number of past returns to include as features, by default 10.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Feature matrix X, target vector y, and the modified DataFrame.
    """
    # Compute daily returns
    df['Return'] = df['Adj Close'].pct_change()
    # Create lagged return features
    for i in range(1, n_lags + 1):
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
    # Drop rows with NaNs resulting from the shift operations
    df = df.dropna().reset_index(drop=True)
    feature_cols = [f'Return_lag_{i}' for i in range(1, n_lags + 1)]
    X = df[feature_cols].values
    y = df['Return'].values
    return X, y, df


def train_test_split_time_series(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    Split feature and target arrays into training and testing sets based on time ordering.

    Unlike typical random splits, time‑series data must be split chronologically.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float, optional
        Proportion of the dataset to allocate to the test set, by default 0.2.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate a regression model on test data.

    Computes mean squared error and directional accuracy. Directional
    accuracy measures the percentage of times the sign of the prediction
    matches the sign of the actual return.

    Parameters
    ----------
    model : object
        A fitted scikit‑learn style estimator with a `predict` method.
    X_test : np.ndarray
        Feature matrix for testing.
    y_test : np.ndarray
        True target values for testing.

    Returns
    -------
    Tuple[float, float, np.ndarray]
        MSE, directional accuracy, and the predicted values.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    # Directional accuracy: fraction of correct sign predictions
    direction_accuracy = np.mean((preds >= 0) == (y_test >= 0))
    return mse, direction_accuracy, preds


def main():
    """Run the full modeling pipeline and write results to disk."""
    data_path = 'data/BTC-USD.csv'
    df = load_data(data_path)
    X, y, df = create_features(df, n_lags=10)
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=0.2)

    # Ensure output directories exist
    import os
    os.makedirs('images', exist_ok=True)

    # Baseline model: Linear Regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    mse_lin, acc_lin, preds_lin = evaluate_model(lin, X_test, y_test)

    # Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    mse_rf, acc_rf, preds_rf = evaluate_model(rf, X_test, y_test)

    # Write results to a text file
    with open('results.txt', 'w') as f:
        f.write('Model evaluation results\n')
        f.write('=========================\n')
        f.write(f'Linear Regression MSE: {mse_lin:.6f}\n')
        f.write(f'Linear Regression Directional Accuracy: {acc_lin:.4f}\n')
        f.write('\n')
        f.write(f'Random Forest MSE: {mse_rf:.6f}\n')
        f.write(f'Random Forest Directional Accuracy: {acc_rf:.4f}\n')

    # Plot cumulative returns for the test period
    cumulative_actual = (1 + y_test).cumprod() - 1
    cumulative_pred_rf = (1 + preds_rf).cumprod() - 1
    plt.figure(figsize=(8, 4))
    # Use the date index from the end of the dataframe corresponding to test period
    test_dates = df['Date'].iloc[-len(y_test):]
    plt.plot(test_dates, cumulative_actual, label='Actual cumulative return')
    plt.plot(test_dates, cumulative_pred_rf, label='RF predicted cumulative return')
    plt.xlabel('Date')
    plt.ylabel('Cumulative return')
    plt.title('Actual vs Predicted Cumulative Returns on Test Set')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/cumulative_returns.png')


if __name__ == '__main__':
    main()
