# Python Stock Analysis

A Python-based stock analysis and prediction tool using data science libraries.

## Setup

1. Create a virtual environment:
   ```
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Verify your installation:
   ```
   python test_environment.py
   ```
   This will check that all required packages are installed correctly.

## Usage

You can use this project in three different ways:

### 1. Python Script

Run the command-line Python script:
```
python stock_analysis.py
```

This will guide you through analyzing a stock, visualizing its price history, and predicting future prices using a linear regression model.

### 2. Web Dashboard

Run the web-based dashboard:
```
python stock_dashboard.py
```

Then open your browser and go to http://127.0.0.1:8050/ to use the interactive dashboard.

### 3. Jupyter Notebook

Open the Jupyter notebook:
```
jupyter notebook Stock_predictor.ipynb
```

## Features

- Stock price data fetching using Yahoo Finance API
- Technical indicators (moving averages, volatility, momentum)
- Linear regression model for price prediction
- Interactive data visualization
- Next-day price prediction
- Web dashboard for easy analysis

## Troubleshooting

If you encounter any issues:

1. Make sure your virtual environment is activated
2. Run `python test_environment.py` to verify all dependencies are installed
3. If any packages are missing, run `pip install -r requirements.txt` again
4. For styling issues with plots, ensure you're using a recent version of matplotlib and seaborn

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance
- statsmodels
- plotly
- dash
- dash-bootstrap-components 