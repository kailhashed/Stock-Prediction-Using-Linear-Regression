#!/usr/bin/env python3
import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create a markdown cell for the title
title_cell = nbf.v4.new_markdown_cell('''# Stock Analysis and Prediction

This notebook contains tools for analyzing stock data and making predictions.''')

# Create a code cell for imports
imports_cell = nbf.v4.new_code_cell('''# Import required libraries
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
sns.set_theme(style="whitegrid")  # Modern seaborn styling''')

# Create a markdown cell for data collection
data_collection_md = nbf.v4.new_markdown_cell('''## Data Collection

We'll use Yahoo Finance to fetch stock data.''')

# Create a code cell for data collection
data_collection_code = nbf.v4.new_code_cell('''# Define stock symbol and date range
stock_symbol = 'AAPL'  # Apple Inc.
start_date = datetime.now() - timedelta(days=365*5)  # 5 years of data
end_date = datetime.now()

# Fetch data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the first few rows
stock_data.head()''')

# Create a markdown cell for data visualization
viz_md = nbf.v4.new_markdown_cell('''## Data Visualization

Let's visualize the stock price history.''')

# Create a code cell for data visualization
viz_code = nbf.v4.new_code_cell('''# Plot closing price
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'])
plt.title(f'{stock_symbol} Stock Price History')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.grid(True)
plt.show()''')

# Create a markdown cell for feature engineering
feature_md = nbf.v4.new_markdown_cell('''## Feature Engineering

Let's create some features for our prediction model.''')

# Create a code cell for feature engineering
feature_code = nbf.v4.new_code_cell('''# Add rolling mean features
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

# Add price momentum (daily returns)
stock_data['Returns'] = stock_data['Close'].pct_change()

# Add volatility (rolling standard deviation)
stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()

# Drop NaN values
stock_data = stock_data.dropna()

# Display the data with new features
stock_data.head()''')

# Create a markdown cell for prediction model
model_md = nbf.v4.new_markdown_cell('''## Prediction Model

Create a simple linear regression model to predict future stock prices.''')

# Create a code cell for prediction model
model_code = nbf.v4.new_code_cell('''# Create features and target
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
print(f'R-squared: {r2:.2f}')''')

# Create a markdown cell for visualization of predictions
pred_viz_md = nbf.v4.new_markdown_cell('''## Visualization of Predictions

Compare actual vs predicted values.''')

# Create a code cell for visualization of predictions
pred_viz_code = nbf.v4.new_code_cell('''# Create a DataFrame with actual and predicted values
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
plt.show()''')

# Create a markdown cell for future predictions
future_md = nbf.v4.new_markdown_cell('''## Future Predictions

Predict the stock price for the next trading day.''')

# Create a code cell for future predictions
future_code = nbf.v4.new_code_cell('''# Get the latest data point
latest_data = stock_data[features].iloc[-1].values.reshape(1, -1)

# Make prediction for the next day
next_day_prediction = model.predict(latest_data)[0]

print(f'Last Close Price: ${stock_data["Close"].iloc[-1]:.2f}')
print(f'Predicted Next Day Price: ${next_day_prediction:.2f}')
print(f'Predicted Change: {((next_day_prediction / stock_data["Close"].iloc[-1]) - 1) * 100:.2f}%')''')

# Add cells to the notebook
nb['cells'] = [
    title_cell,
    imports_cell,
    data_collection_md,
    data_collection_code,
    viz_md,
    viz_code,
    feature_md,
    feature_code,
    model_md,
    model_code,
    pred_viz_md,
    pred_viz_code,
    future_md,
    future_code
]

# Write the notebook to a file
with open('Stock_predictor.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 'Stock_predictor.ipynb' created successfully!") 