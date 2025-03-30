#!/usr/bin/env python3
# Stock Analysis and Prediction Script

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set visualization styles
plt.style.use('ggplot')  # Using a stable, non-deprecated style
sns.set_theme(style="whitegrid")  # Modern seaborn styling

def fetch_stock_data(stock_symbol, years=5):
    """Fetch stock data from Yahoo Finance."""
    try:
        start_date = datetime.now() - timedelta(days=365*years)
        end_date = datetime.now()
        
        print(f"Fetching data for {stock_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {stock_symbol}")
            
        print(f"Fetched {len(stock_data)} days of data.")
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        raise

def visualize_stock_price(stock_data, stock_symbol):
    """Visualize the stock price history."""
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'])
    plt.title(f'{stock_symbol} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.grid(True)
    plt.show()

def create_features(stock_data):
    """Create features for prediction model."""
    # Add rolling mean features
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    
    # Add price momentum (daily returns)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Add volatility (rolling standard deviation)
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
    
    # Drop NaN values
    stock_data = stock_data.dropna()
    
    return stock_data

def build_model(stock_data):
    """Build and train a prediction model."""
    # Create features and target
    # Predict next day's closing price
    stock_data['Target'] = stock_data['Close'].shift(-1)
    stock_data = stock_data.dropna()
    
    # Select features
    features = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility']
    X = stock_data[features]
    y = stock_data['Target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    
    return model, features, X_test, y_test, predictions, stock_data

def visualize_predictions(stock_symbol, y_test, predictions):
    """Visualize actual vs predicted values."""
    # Create a DataFrame with actual and predicted values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    results = results.sort_index()
    
    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['Actual'], label='Actual')
    plt.plot(results.index, results['Predicted'], label='Predicted', alpha=0.7)
    plt.title(f'{stock_symbol} - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def safe_series_to_float(series_value):
    """Safely convert a pandas Series value to float without triggering deprecation warnings."""
    if hasattr(series_value, 'iloc'):
        return float(series_value.iloc[0]) if len(series_value) == 1 else float(series_value)
    return float(series_value)

def predict_next_day(model, features, stock_data):
    """Predict the stock price for the next trading day."""
    # Get the latest data point
    latest_data = stock_data[features].iloc[-1].values.reshape(1, -1)
    
    # Make prediction for the next day
    next_day_prediction = model.predict(latest_data)[0]
    
    # Get the last close price as a scalar value
    last_close = safe_series_to_float(stock_data["Close"].iloc[-1])
    
    print(f'Last Close Price: ${last_close:.2f}')
    print(f'Predicted Next Day Price: ${next_day_prediction:.2f}')
    print(f'Predicted Change: {((next_day_prediction / last_close) - 1) * 100:.2f}%')
    
    return next_day_prediction

def main():
    # Set the stock symbol
    stock_symbol = input("Enter stock symbol (default: AAPL): ") or "AAPL"
    
    # Fetch data
    stock_data = fetch_stock_data(stock_symbol)
    
    # Visualize data
    visualize_stock_price(stock_data, stock_symbol)
    
    # Create features
    stock_data = create_features(stock_data)
    
    # Build and evaluate the model
    model, features, X_test, y_test, predictions, stock_data = build_model(stock_data)
    
    # Visualize predictions
    visualize_predictions(stock_symbol, y_test, predictions)
    
    # Predict next day's price
    predict_next_day(model, features, stock_data)

if __name__ == "__main__":
    main() 