#!/usr/bin/env python3
# Stock Analysis Dashboard using Dash

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def safe_series_to_float(series_value):
    """Safely convert a pandas Series value to float without triggering deprecation warnings."""
    if hasattr(series_value, 'iloc'):
        return float(series_value.iloc[0]) if len(series_value) == 1 else float(series_value)
    return float(series_value)

def fetch_stock_data(stock_symbol, years=5):
    """Fetch stock data from Yahoo Finance."""
    try:
        start_date = datetime.now() - timedelta(days=365*years)
        end_date = datetime.now()
        
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {stock_symbol}")
            
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        raise

def create_features(stock_data):
    """Create features for prediction model."""
    # Add rolling mean features
    df = stock_data.copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Add price momentum (daily returns)
    df['Returns'] = df['Close'].pct_change()
    
    # Add volatility (rolling standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def build_model(stock_data):
    """Build and train a prediction model."""
    # Create a copy of the data
    df = stock_data.copy()
    
    # Create features and target
    # Predict next day's closing price
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    # Select features
    features = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility']
    X = df[features]
    y = df['Target']
    
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
    
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    results_df = results_df.sort_index()
    
    # Next day prediction
    latest_data = df[features].iloc[-1].values.reshape(1, -1)
    next_day_prediction = model.predict(latest_data)[0]
    
    # Get the last closing price safely
    last_close = safe_series_to_float(df["Close"].iloc[-1])
    
    return {
        'model_metrics': {'mse': mse, 'r2': r2},
        'results_df': results_df,
        'next_day_prediction': next_day_prediction,
        'last_close': last_close
    }

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Stock Analysis Dashboard", className="text-center my-4"),
            html.P("Enter a stock symbol to analyze and predict its price", className="text-center"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id="stock-input", placeholder="Enter stock symbol (e.g., AAPL)", type="text", value="AAPL"),
                dbc.InputGroupText("Years of data:"),
                dbc.Select(
                    id="years-input",
                    options=[
                        {"label": "1 year", "value": "1"},
                        {"label": "3 years", "value": "3"},
                        {"label": "5 years", "value": "5"},
                        {"label": "10 years", "value": "10"},
                    ],
                    value="5",
                ),
                dbc.Button("Analyze", id="analyze-button", color="primary"),
            ]),
            html.Div(id="error-message", className="text-danger mt-2"),
        ], width={"size": 8, "offset": 2}),
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                color="#17BECF",
                children=[
                    html.Div(id="stock-data-container", className="mt-4")
                ]
            )
        ])
    ]),
    
], fluid=True)

@app.callback(
    [Output("stock-data-container", "children"),
     Output("error-message", "children")],
    Input("analyze-button", "n_clicks"),
    State("stock-input", "value"),
    State("years-input", "value"),
    prevent_initial_call=True
)
def update_stock_analysis(n_clicks, stock_symbol, years):
    if not stock_symbol:
        return None, "Please enter a valid stock symbol"
    
    try:
        # Clear error message
        error_msg = ""
        
        # Fetch data
        stock_data = fetch_stock_data(stock_symbol, int(years))
        
        if stock_data.empty:
            return None, f"No data found for {stock_symbol}"
        
        # Create price chart
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#17BECF')
        ))
        price_fig.update_layout(
            title=f'{stock_symbol} Stock Price History',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=500
        )
        
        # Create features and build model
        df_with_features = create_features(stock_data)
        model_results = build_model(df_with_features)
        
        # Create prediction chart
        results_df = model_results['results_df']
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=results_df.index, 
            y=results_df['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#17BECF')
        ))
        pred_fig.add_trace(go.Scatter(
            x=results_df.index, 
            y=results_df['Predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='#FF9900')
        ))
        pred_fig.update_layout(
            title=f'{stock_symbol} - Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=500
        )
        
        # Calculate prediction metrics
        last_close = model_results['last_close']
        next_day_prediction = model_results['next_day_prediction']
        change_percentage = ((next_day_prediction / last_close) - 1) * 100
        
        # Display metrics
        content = [
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4(f"{stock_symbol} Summary", className="card-title"),
                            html.P(f"Data from {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}"),
                            html.P(f"Total trading days: {len(stock_data)}"),
                            html.P(f"Last close price: ${last_close:.2f}"),
                            html.P([
                                "Predicted next day price: ",
                                html.Span(
                                    f"${next_day_prediction:.2f} ({change_percentage:.2f}%)",
                                    style={"color": "green" if change_percentage > 0 else "red"}
                                )
                            ]),
                            html.P([
                                "Model accuracy: ",
                                html.Span(f"R² score: {model_results['model_metrics']['r2']:.4f}"),
                                html.Br(),
                                html.Span(f"Mean squared error: {model_results['model_metrics']['mse']:.4f}")
                            ]),
                        ])
                    )
                ], width=12)
            ]),
            html.H3("Stock Price History", className="mt-4"),
            dcc.Graph(figure=price_fig),
            html.H3("Prediction Analysis", className="mt-4"),
            dcc.Graph(figure=pred_fig)
        ]
        
        return content, error_msg
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, f"Error analyzing {stock_symbol}: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=8050) 