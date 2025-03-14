import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

# Streamlit App Title
st.title("📈 Pair Trading Analysis App")

# User Input: Multiple Stock Pairs
st.sidebar.header("Enter Stock Pairs")

num_pairs = st.sidebar.number_input("How many pairs do you want to analyze?", min_value=1, max_value=10, value=1)

stock_pairs = []
for i in range(num_pairs):
    stock_a = st.sidebar.text_input(f"Stock A (Pair {i+1})", key=f"stock_a_{i}")
    stock_b = st.sidebar.text_input(f"Stock B (Pair {i+1})", key=f"stock_b_{i}")
    
    if stock_a and stock_b:
        stock_pairs.append((stock_a.upper(), stock_b.upper()))

# User Input: Date Range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

if not stock_pairs:
    st.warning("Enter at least one valid stock pair to analyze.")
    st.stop()

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch stock data
stock_data = {}
for stock_a, stock_b in stock_pairs:
    stock_data[stock_a] = get_stock_data(stock_a)
    stock_data[stock_b] = get_stock_data(stock_b)

# Pair Trading Analysis
output_data = []

for stock_a, stock_b in stock_pairs:
    if stock_data[stock_a] is None or stock_data[stock_b] is None:
        continue
    
    # Align data to same date range
    df = pd.DataFrame({stock_a: stock_data[stock_a], stock_b: stock_data[stock_b]}).dropna()

    # Regression 1 (Stock A ~ Stock B)
    X = sm.add_constant(df[stock_b])
    y = df[stock_a]
    model1 = sm.OLS(y, X).fit()
    resid1 = model1.resid
    se1 = np.std(resid1) / (model1.bse[0] if len(model1.bse) > 0 else np.nan)

    # Regression 2 (Stock B ~ Stock A)
    X = sm.add_constant(df[stock_a])
    y = df[stock_b]
    model2 = sm.OLS(y, X).fit()
    resid2 = model2.resid
    se2 = np.std(resid2) / (model2.bse[0] if len(model2.bse) > 0 else np.nan)

    # Choose best model
    if se1 < se2:
        best_beta, best_intercept, best_resid = model1.params[1], model1.params[0], resid1
        best_se = se1
    else:
        best_beta, best_intercept, best_resid = model2.params[1], model2.params[0], resid2
        best_se = se2

    # ADF Test
    if best_resid.isnull().any() or len(best_resid) < 10:
        adf_test_value = np.nan
    else:
        try:
            adf_test_value = adfuller(best_resid.dropna())[0]
        except ValueError:
            adf_test_value = np.nan

    # Current Residual
    current_residual = best_resid.iloc[-1] if len(best_resid) > 0 else np.nan

    # Store results
    output_data.append([stock_a, stock_b, best_beta, best_intercept, adf_test_value, best_se, current_residual])

# Display Results
st.write("## Pair Trading Analysis Results")
output_df = pd.DataFrame(output_data, columns=["Stock A", "Stock B", "Beta", "Intercept", "ADF Test Value", "STD ERROR", "Current Residual"])
st.dataframe(output_df)

# Download Option
st.download_button(label="Download Results as CSV", data=output_df.to_csv(index=False), file_name="pair_trading_results.csv", mime="text/csv")
