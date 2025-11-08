# app.py -- Explicit calculations shown (TSS/ESS/SSR/F/Prob(F)/RMSE) + coef table + ANOVA
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import io
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="MultiRegressor — explicit stats", layout="wide")
st.title("MultiRegressor — Regression with explicit calculated stats")

# --- upload ---
uploaded_file = st.file_uploader("Upload CSV / XLS / XLSX", type=["csv","xls","xlsx"])
if not uploaded_file:
    st.info("Upload a file")
    st.stop()

# --- read file (sheet choice for excel) ---
name = uploaded_file.name.lower()
try:
    if name.endswith(".csv"):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    else:
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Choose sheet", xls.sheet_names)
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(200))

# --- auto-detect / controls ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# default heuristics
default_y = numeric_cols[0] if numeric_cols else None
default_xs = [c for c in numeric_cols if c != default_y]

st.sidebar.header("Regression controls")
dep = st.sidebar.selectbox("Dependent (Y)", options=["(auto)"] + all_cols, index=0)
if dep == "(auto)":
    dep = default_y
preds = st.sidebar.multiselect("Predictors (X)", options=[c for c in all_cols if c != dep], default=default_xs)
intercept = st.sidebar.checkbox("Include intercept", value=True)
drop_na = st.sidebar.checkbox("Drop rows with missing values", value=True)
run_btn = st.sidebar.button("Run regression")

if not dep or not preds:
    st.warning("Set Y and at least one X in the sidebar.")
    st.stop()

# --- prepare data ---
data = df[[dep] + preds].copy()
# coerce numeric (so strings become NaN)
data = data.apply(pd.to_numeric, errors='coerce')
if drop_na:
    data = data.dropna()
if data.shape[0] == 0:
    st.error("No usable rows after coercion / dropna.")
    st.stop()

y = data[dep].values.astype(float)
X = data[preds].astype(float)
if intercept:
    X = sm.add_constant(X, has_constant='add')

n = len(y)
p = X.shape[1]  # includes intercept if present
df_model = p - 1  # number of predictors (excluding constant)
df_resid = n - p

if n <= p:
    st.error(f"Not enough observations: n={n}, parameters={p}. Need n > p.")
    st.stop()

if run_btn or True:  # auto-run (always run when page loads)
    # fit using statsmodels (we still need b, se, t, p)
    model = sm.OLS(y, X)
    results = model.fit()

    # Coeff table
    params = results.params
    bse = results.bse
    tvals = results.tvalues
    pvals = results.pvalues
    conf = results.conf_int(alpha=0.05)
    conf_df = pd.DataFrame(conf, columns=['CI_lower','CI_upper'], index=params.index)

    coef_table = pd.DataFrame({
        'Variable': params.index,
        'Coef': params.values,
        'Std Err': bse.values,
        't': tvals.values,
        'p-value': pvals.values
    }).set_index('Variable')
    coef_table[['CI_lower','CI_upper']] = conf_df
    # significance stars
    def stars(p):
        if pd.isna(p): return ''
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        if p < 0.1: return '.'
        return ''
    coef_table['signif'] = coef_table['p-value'].apply(stars)

    # --- Explicit calculations (do not rely solely on statsmodels internals) ---
    y_mean = y.mean()
    # Predictions and residuals
    y_hat = results.predict(X)
    resid = y - y_hat

    # TSS (total sum squares), centered around mean
    TSS = np.sum((y - y_mean)**2)
    # SSR (residual sum of squares)
    SSR = np.sum((resid)**2)
    # ESS (explained sum of squares)
    ESS = np.sum((y_hat - y_mean)**2)
    # check decomposition
    # TSS == ESS + SSR (within numerical tolerance)
    decomposition_check = TSS - (ESS + SSR)

    # Mean squares
    MSR = ESS / df_model if df_model > 0 else np.nan  # model mean square
    MSE = SSR / df_resid

    # F-stat (explicit)
    F_stat = MSR / MSE if (not np.isnan(MSR) and MSE > 0) else np.nan
    # Prob(F) (right-tail)
    Prob_F = stats.f.sf(F_stat, df_model, df_resid) if not np.isnan(F_stat) else np.nan

    # R-squared and adjusted
    R2 = ESS / TSS if TSS > 0 else np.nan
    AdjR2 = 1 - (1 - R2) * (n - 1) / df_resid if not np.isnan(R2) else np.nan

    # RMSE
    RMSE = np.sqrt(MSE)

    # AIC/BIC from results
    AIC = results.aic
    BIC = results.bic

    # --- Build ANOVA-like table (Model vs Residual) ---
    anova_df = pd.DataFrame([
        {'Source': 'Model', 'df': int(df_model), 'sum_sq': ESS, 'mean_sq': MSR, 'F': F_stat, 'PR(>F)': Prob_F},
        {'Source': 'Residual', 'df': int(df_resid), 'sum_sq': SSR, 'mean_sq': MSE, 'F': np.nan, 'PR(>F)': np.nan},
        {'Source': 'Total', 'df': int(n - 1), 'sum_sq': TSS, 'mean_sq': np.nan, 'F': np.nan, 'PR(>F)': np.nan}
    ])

    # --- Show everything clearly ---
    st.subheader("Coefficient table")
    st.dataframe(coef_table.round(6).reset_index())

    st.markdown("**Model-level statistics (explicitly calculated):**")
    stats_items = [
        ("Observations (n)", n),
        ("Number of parameters (p)", p),
        ("Degrees of freedom (model)", df_model),
        ("Degrees of freedom (resid)", df_resid),
        ("TSS (total sum squares)", TSS),
        ("ESS (explained sum squares)", ESS),
        ("SSR (residual sum squares)", SSR),
        ("TSS - (ESS+SSR) (should be ~0)", decomposition_check),
        ("MSR (ESS/df_model)", MSR),
        ("MSE (SSR/df_resid)", MSE),
        ("F-statistic (MSR/MSE)", F_stat),
        ("Prob (F-statistic) (right-tail)", Prob_F),
        ("R-squared", R2),
        ("Adjusted R-squared", AdjR2),
        ("RMSE (sqrt(MSE))", RMSE),
        ("AIC (from model)", AIC),
        ("BIC (from model)", BIC)
    ]
    stats_df = pd.DataFrame(stats_items, columns=['Stat','Value'])
    # format numbers
    def fmt_val(v):
        if isinstance(v, (int,np.integer)):
            return str(v)
        if isinstance(v, (float,np.floating)):
            return f"{v:.8g}"
        return str(v)
    stats_df['Value'] = stats_df['Value'].apply(fmt_val)
    st.table(stats_df)

    st.subheader("ANOVA (Model vs Residual vs Total)")
    st.dataframe(anova_df.round(6))

    # --- diagnostic plots ---
    st.subheader("Diagnostic plots")
    c1, c2 = st.columns(2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_hat, resid, alpha=0.6)
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    c1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    stats.probplot(resid, dist='norm', plot=ax2)
    ax2.set_title("QQ plot of residuals")
    c2.pyplot(fig2)

    # --- download outputs ---
    out_all = data.copy()
    out_all['_y_true'] = y
    out_all['_y_pred'] = y_hat
    out_all['_resid'] = resid

    # coefficients CSV
    buf = io.StringIO()
    coef_table.reset_index().to_csv(buf, index=False)
    st.download_button("Download coefficients (CSV)", data=buf.getvalue(), file_name="coefficients.csv", mime="text/csv")

    # full XLSX
    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
        coef_table.reset_index().to_excel(writer, sheet_name='Coefficients', index=False)
        stats_df.to_excel(writer, sheet_name='ModelStats', index=False)
        anova_df.to_excel(writer, sheet_name='ANOVA', index=False)
        out_all.reset_index().to_excel(writer, sheet_name='Predictions', index=False)
    buf_xlsx.seek(0)
    st.download_button("Download full regression output (XLSX)", data=buf_xlsx, file_name='regression_output.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # full textual summary
    with st.expander("statsmodels summary (text)"):
        st.text(results.summary().as_text())
