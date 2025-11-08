# app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import io
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="MultiRegressor (Streamlit)", layout="wide")

st.title("Multi Regressor — upload data & get OLS table")

st.markdown(
    """
Upload a CSV or Excel file. Choose the dependent variable and one or more predictors,
then press **Run regression**. Numeric-only rows are used (rows with missing/NaN values
in selected columns are dropped).
"""
)

# --- File upload ---
uploaded_file = st.file_uploader(
    "Upload CSV / XLS / XLSX", type=["csv", "xls", "xlsx"], accept_multiple_files=False
)

df = None
sheet_names = []
selected_sheet = None

if uploaded_file:
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            sheet_names = ["CSV"]
            selected_sheet = "CSV"
        else:
            # Excel
            uploaded_file.seek(0)
            # read sheet names
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            selected_sheet = st.selectbox("Select sheet (Excel)", sheet_names, index=0)
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Data preview ---
if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(200))

    # auto-detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    st.sidebar.header("Regression settings")
    dep = st.sidebar.selectbox("Dependent variable (Y)", options=[""] + all_cols)
    # Predictors multi-select
    preds = st.sidebar.multiselect("Predictors (X) — pick one or more", options=[c for c in all_cols if c != dep])

    intercept = st.sidebar.checkbox("Include intercept (constant)", value=True)
    add_interactions = st.sidebar.checkbox("Automatically add simple pairwise interaction terms (X1*X2)", value=False)
    drop_na_rows = st.sidebar.checkbox("Drop rows with NaNs in selected columns", value=True)

    run = st.sidebar.button("Run regression")

    # Optionally allow simple transformations
    st.sidebar.markdown("**Optional transforms**")
    transform_cols = st.sidebar.multiselect("Columns to log-transform (natural log). Non-positive values will be removed.", options=all_cols)

    if run:
        if not dep:
            st.sidebar.error("Pick a dependent variable (Y).")
        elif not preds:
            st.sidebar.error("Pick at least one predictor.")
        else:
            # prepare data copy
            data = df.copy()

            # apply log transforms
            if transform_cols:
                for c in transform_cols:
                    # replace badly-behaving values -> drop later
                    data[f"log_{c}"] = np.where(data[c] > 0, np.log(data[c]), np.nan)
                    # if user selected transform column in predictors or dep, replace
                    if c == dep:
                        dep = f"log_{c}"
                    data.columns = data.columns  # no-op to keep things consistent

            # build X, optionally add interactions
            X = data[preds].copy()
            # if any transform used that created new column names, let user choose them explicitly next run — but for simplicity we won't auto-swap predictors here

            if add_interactions:
                # add pairwise interactions for predictors
                new_cols = {}
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        a = preds[i]
                        b = preds[j]
                        colname = f"{a}__x__{b}"
                        new_cols[colname] = data[a] * data[b]
                if new_cols:
                    X = pd.concat([X, pd.DataFrame(new_cols)], axis=1)

            # If user asked to drop rows with NaNs in selected columns
            selected_cols = [dep] + list(X.columns)
            if drop_na_rows:
                data2 = data[selected_cols].dropna()
            else:
                data2 = data[selected_cols]

            if data2.shape[0] < (len(X.columns) + 1):
                st.error("Not enough observations after dropping missing values (need more rows than parameters).")
            else:
                y = data2[dep]
                X_final = data2[X.columns]

                if intercept:
                    X_final = sm.add_constant(X_final, has_constant='add')

                # Ensure numeric
                try:
                    X_final = X_final.astype(float)
                    y = y.astype(float)
                except Exception as e:
                    st.error(f"Non-numeric data found in selected columns: {e}")
                    st.stop()

                # Fit OLS
                model = sm.OLS(y, X_final)
                results = model.fit()

                # Regression output
                st.header("Regression output")
                coef_df = results.summary2().tables[1].reset_index()
                coef_df = coef_df.rename(columns={"index": "Variable"})
                st.dataframe(coef_df.style.format("{:.6g}"))

                # Key stats
                st.write("**Model fit:**")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.write(f"Observations (n): **{int(results.nobs)}**")
                    st.write(f"R²: **{results.rsquared:.6f}**")
                    st.write(f"Adjusted R²: **{results.rsquared_adj:.6f}**")
                with stats_col2:
                    rss = np.sum(results.resid ** 2)
                    mse = results.mse_resid
                    rmse = np.sqrt(mse)
                    st.write(f"RSS: **{rss:.6f}**")
                    st.write(f"MSE (resid): **{mse:.6f}**")
                    st.write(f"RMSE: **{rmse:.6f}**")

                # Predictions and residuals table
                preds_series = results.predict(X_final)
                resid = results.resid
                out_df = data2.copy()
                out_df["_yhat"] = preds_series
                out_df["_residual"] = resid

                # Download regression table (coef)
                buf = io.StringIO()
                coef_df.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download coefficients (CSV)", data=buf, file_name="regression_coefficients.csv", mime="text/csv")

                # Download predictions + residuals
                buf2 = io.BytesIO()
                out_df.to_excel(buf2, index=False, engine="openpyxl")
                buf2.seek(0)
                st.download_button("Download predictions & residuals (XLSX)", data=buf2, file_name="predictions_residuals.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # Plots: Residuals vs Fitted
                st.subheader("Diagnostic plots")
                fig1, ax1 = plt.subplots()
                ax1.scatter(preds_series, resid, alpha=0.6)
                ax1.axhline(0, color="k", linestyle="--", linewidth=0.8)
                ax1.set_xlabel("Fitted values")
                ax1.set_ylabel("Residuals")
                ax1.set_title("Residuals vs Fitted")
                st.pyplot(fig1)

                # QQ plot
                fig2, ax2 = plt.subplots()
                stats.probplot(resid, dist="norm", plot=ax2)
                ax2.set_title("QQ plot of residuals")
                st.pyplot(fig2)

                # Show summary text
                with st.expander("Full regression summary (text)"):
                    st.text(results.summary().as_text())

else:
    st.info("Upload a CSV or Excel file to begin.")
