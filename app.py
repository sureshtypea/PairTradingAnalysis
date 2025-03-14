import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import io

st.title("Pair Trading Analysis")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file (Stock Data & Pairs)", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    xls = pd.ExcelFile(uploaded_file)

    # Read stock pairs from "Pairs" sheet
    final_pairs = pd.read_excel(xls, sheet_name="Pairs")
    final_pairs.columns = final_pairs.columns.str.strip()  # Clean column names

    st.write("**Uploaded Stock Pairs:**")
    st.dataframe(final_pairs)

    results = []

    # Function to read stock data from Excel sheets
    def load_stock_data(sheet_name):
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            close_col = [col for col in df.columns if "Close" in col]
            if not close_col:
                return None
            close_col = close_col[0]  # Use the first match
            df.index = range(len(df))  # Reset index for serial alignment
            return df[close_col]
        except Exception as e:
            st.warning(f"Error reading sheet {sheet_name}: {e}")
            return None

    for _, row in final_pairs.iterrows():
        stock_A, stock_B = row.iloc[0], row.iloc[1]
        
        close_A = load_stock_data(stock_A)
        close_B = load_stock_data(stock_B)
        
        if close_A is None or close_B is None:
            st.warning(f"Missing data for {stock_A} or {stock_B}. Skipping...")
            continue

        # Align data lengths
        min_length = min(len(close_A), len(close_B))
        close_A, close_B = close_A.iloc[:min_length], close_B.iloc[:min_length]

        # Create DataFrame
        merged_df = pd.DataFrame({stock_A: close_A.values, stock_B: close_B.values})

        # Perform Regression: A = F(B)
        X = sm.add_constant(merged_df[stock_B])
        y = merged_df[stock_A]
        model1 = sm.OLS(y, X).fit()
        resid1 = model1.resid
        tg1 = np.std(resid1)
        gt1 = model1.bse[0] if len(model1.bse) > 0 else np.nan
        se1 = gt1 / tg1 if tg1 != 0 else np.inf

        # Perform Regression: B = F(A)
        X = sm.add_constant(merged_df[stock_A])
        y = merged_df[stock_B]
        model2 = sm.OLS(y, X).fit()
        resid2 = model2.resid
        tg2 = np.std(resid2)
        gt2 = model2.bse[0] if len(model2.bse) > 0 else np.nan
        se2 = gt2 / tg2 if tg2 != 0 else np.inf

        # Select the best model based on lowest standard error
        if se1 < se2:
            best_beta, best_intercept = model1.params[1], model1.params[0]
            best_se, best_resid = se1, resid1
        else:
            best_beta, best_intercept = model2.params[1], model2.params[0]
            best_se, best_resid = se2, resid2

        # Perform ADF Test on residuals
        adf_test_value = adfuller(best_resid)[0]

        # Store latest residual value
        current_residual = best_resid.iloc[-1]

        # Store results
        results.append([stock_A, stock_B, best_beta, best_intercept, adf_test_value, best_se, current_residual])

    # Display results in table format
    st.write("**Pair Trading Analysis Results:**")
    results_df = pd.DataFrame(results, columns=["Stock A", "Stock B", "Beta", "Intercept", "ADF Test Value", "STD ERROR", "Current Residual"])
    st.dataframe(results_df)

    # Create downloadable CSV file
    csv_output = io.StringIO()
    results_df.to_csv(csv_output, index=False)
    csv_output.seek(0)

    st.download_button(
        label="Download Results as CSV",
        data=csv_output,
        file_name="pair_trading_results.csv",
        mime="text/csv"
    )
