import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Streamlit App Title
st.title("Pair Trading Analysis")

# 📂 Upload the single Excel File
st.sidebar.header("Upload the Pair Trading Data File")
uploaded_file = st.sidebar.file_uploader("Upload Template.xlsx", type=["xlsx"])

if not uploaded_file:
    st.warning("Please upload the Excel file to proceed.")
    st.stop()

# Read Excel File
xls = pd.ExcelFile(uploaded_file)

# Load the "Pairs" sheet
pairs_df = xls.parse("Pairs")
pairs_df.columns = pairs_df.columns.str.strip()

st.write("### Uploaded Stock Pairs Data")
st.dataframe(pairs_df)

# Dictionary to store stock data
stock_data = {}

# Load stock closing prices from each sheet
for sheet_name in xls.sheet_names:
    if sheet_name == "Pairs":
        continue  # Skip the pairs sheet

    df = xls.parse(sheet_name)
    
    # Ensure required columns exist
    if "Close" not in df.columns:
        st.warning(f"Skipping {sheet_name}: No 'Close' column found.")
        continue

    df = df[["Close"]].dropna()  # Keep only Close prices
    df.index = range(len(df))  # Reset index to ensure proper alignment
    stock_data[sheet_name] = df

st.success("Files uploaded and processed successfully!")

# 📈 Pair Trading Analysis
output_data = []

for index, row in pairs_df.iterrows():
    stock_A, stock_B = row.iloc[0], row.iloc[1]  # Assuming first two columns contain stock names

    if stock_A not in stock_data or stock_B not in stock_data:
        st.warning(f"Skipping pair ({stock_A}, {stock_B}) - Data missing.")
        continue

    close_A = stock_data[stock_A]["Close"]
    close_B = stock_data[stock_B]["Close"]

    # Ensure both stocks have the same number of data points
    min_length = min(len(close_A), len(close_B))
    close_A, close_B = close_A.iloc[:min_length], close_B.iloc[:min_length]

    # Merge the data
    merged_df = pd.DataFrame({stock_A: close_A.values, stock_B: close_B.values})

    # Run Regression 1 (Stock A ~ Stock B)
    X = sm.add_constant(merged_df[stock_B])
    y = merged_df[stock_A]
    model1 = sm.OLS(y, X).fit()
    resid1 = model1.resid
    se1 = np.std(resid1) / (model1.bse[0] if len(model1.bse) > 0 else np.nan)

    # Run Regression 2 (Stock B ~ Stock A)
    X = sm.add_constant(merged_df[stock_A])
    y = merged_df[stock_B]
    model2 = sm.OLS(y, X).fit()
    resid2 = model2.resid
    se2 = np.std(resid2) / (model2.bse[0] if len(model2.bse) > 0 else np.nan)

    # Select the best model (lower standard error)
    if se1 < se2:
        best_beta, best_intercept, best_resid = model1.params[1], model1.params[0], resid1
        best_se = se1
    else:
        best_beta, best_intercept, best_resid = model2.params[1], model2.params[0], resid2
        best_se = se2

    # ADF Test for Stationarity
    if best_resid.isnull().any() or len(best_resid) < 10:
        adf_test_value = np.nan
    else:
        try:
            adf_test_value = adfuller(best_resid.dropna())[0]
        except ValueError:
            adf_test_value = np.nan

    # Current Residual Value
    current_residual = best_resid.iloc[-1] if len(best_resid) > 0 else np.nan

    # Store output
    output_data.append([stock_A, stock_B, best_beta, best_intercept, adf_test_value, best_se, current_residual])

# Display Results in Streamlit
st.write("## Pair Trading Analysis Results")
output_df = pd.DataFrame(output_data, columns=["Stock A", "Stock B", "Beta", "Intercept", "ADF Test Value", "STD ERROR", "Current Residual"])
st.dataframe(output_df)

# Downloadable CSV
st.download_button(label="Download Results as CSV", data=output_df.to_csv(index=False), file_name="pair_trading_results.csv", mime="text/csv")
