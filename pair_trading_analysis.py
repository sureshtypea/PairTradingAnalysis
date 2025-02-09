import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import tabulate

# Define the folder path where all files are stored
folder_path = r"C:\Users\sures\Pairtrading\Files"  # Update with actual folder path


final_pairs_file = os.path.join(folder_path, "finalpair.xlsx")
final_pairs = pd.read_excel(final_pairs_file)
final_pairs.columns = final_pairs.columns.str.strip()


print("Columns in finalpair.xlsx:", final_pairs.columns.tolist())


output_data = []


def load_stock_data(stock_name):
    file_path = os.path.join(folder_path, f"{stock_name}.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    
    print(f"Columns in {stock_name}.csv:", df.columns.tolist())

    
    close_col = [col for col in df.columns if "Close" in col]
    if not close_col:
        print(f"Missing 'Close' column in {stock_name}.csv")
        return None
    close_col = close_col[0]  # Use the first match

    
    df.index = range(len(df))  # Reset index to ensure serial alignment
    return df[close_col]  # Return Close prices only


for index, row in final_pairs.iterrows():
    
    stock_A, stock_B = row.iloc[0], row.iloc[1]  # Assuming first two columns contain stock names

    
    close_A = load_stock_data(stock_A)
    close_B = load_stock_data(stock_B)
    if close_A is None or close_B is None:
        continue

    
    min_length = min(len(close_A), len(close_B))
    close_A, close_B = close_A.iloc[:min_length], close_B.iloc[:min_length]

    
    merged_df = pd.DataFrame({stock_A: close_A.values, stock_B: close_B.values})

    
    X = sm.add_constant(merged_df[stock_B])
    y = merged_df[stock_A]
    model1 = sm.OLS(y, X).fit()
    resid1 = model1.resid
    tg1 = np.std(resid1)
    gt1 = model1.bse[0] if len(model1.bse) > 0 else np.nan
    se1 = gt1 / tg1 if tg1 != 0 else np.inf

    
    X = sm.add_constant(merged_df[stock_A])
    y = merged_df[stock_B]
    model2 = sm.OLS(y, X).fit()
    resid2 = model2.resid
    tg2 = np.std(resid2)
    gt2 = model2.bse[0] if len(model2.bse) > 0 else np.nan
    se2 = gt2 / tg2 if tg2 != 0 else np.inf

    
    if se1 < se2:
        best_beta, best_intercept = model1.params[1], model1.params[0]
        best_se, best_resid = se1, resid1
    else:
        best_beta, best_intercept = model2.params[1], model2.params[0]
        best_se, best_resid = se2, resid2

    
    adf_test_value = adfuller(best_resid)[0]

    
    current_residual = best_resid.iloc[-1]

    
    output_data.append([
        stock_A, stock_B, best_beta, best_intercept, adf_test_value, best_se, current_residual
    ])


print("\nPair Trading Analysis Results:\n")
print(tabulate.tabulate(output_data, headers=[
    "Stock A", "Stock B", "Beta", "Intercept", "ADF Test Value", "STD ERROR", "Current Residual"
], tablefmt="fancy_grid"))
