# app.py -- Robust Sheet1-style OLS with safe column checks and ANOVA + model stats
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import io
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="MultiRegressor — Robust Sheet1-style", layout="wide")
st.title("MultiRegressor — Robust Sheet1-style OLS")

# ---------------- File upload ----------------
uploaded_file = st.file_uploader("Upload CSV / XLS / XLSX", type=["csv", "xls", "xlsx"])
if not uploaded_file:
    st.info("Upload a file to begin.")
    st.stop()

# ---------------- Read file + choose sheet ----------------
file_name = uploaded_file.name.lower()
try:
    if file_name.endswith(".csv"):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    else:
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file)
        sheet_choice = st.selectbox("Select sheet (Excel)", xls.sheet_names)
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_choice)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(200))

# ---------------- Auto-detect numeric columns and defaults ----------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# Heuristic for Y
preferred_y_names = ['y','target','dependent','outcome','response']
auto_y = None
for n in preferred_y_names:
    for col in all_cols:
        if col.strip().lower() == n:
            auto_y = col
            break
    if auto_y:
        break
if not auto_y and numeric_cols:
    auto_y = numeric_cols[0]  # safest fallback

auto_xs = [c for c in numeric_cols if c != auto_y]

# ---------------- Sidebar controls ----------------
st.sidebar.header("Regression settings")
dep = st.sidebar.selectbox("Dependent (Y) — override", options=["(auto)"] + all_cols, index=0)
if dep == "(auto)":
    dep = auto_y

preds = st.sidebar.multiselect("Predictors (X) — override", options=[c for c in all_cols if c != dep],
                              default=auto_xs if len(auto_xs)>0 else None)

intercept = st.sidebar.checkbox("Include intercept (constant)", value=True)
drop_na_rows = st.sidebar.checkbox("Drop rows with missing values in selected columns", value=True)
add_interactions = st.sidebar.checkbox("Add pairwise interactions (X1*X2)", value=False)
auto_run = st.sidebar.checkbox("Auto-run", value=True)
manual_run = st.sidebar.button("Run regression (manual)")

# If user didn't pick preds, but auto_xs exist, set to auto_xs
if (not preds or len(preds)==0) and auto_xs:
    preds = auto_xs

# -------------- Validate columns exist ----------------
missing_cols = []
if dep is None or dep not in df.columns:
    st.error(f"Dependent variable not found in data (selected: {dep}). Pick a different Y in the sidebar.")
    st.stop()

if not preds or len(preds) == 0:
    st.error("No predictors selected. Pick at least one predictor column in the sidebar.")
    st.stop()

for c in preds:
    if c not in df.columns:
        missing_cols.append(c)

if missing_cols:
    st.error(f"The following predictors aren't in the uploaded data: {missing_cols}. Please pick valid columns.")
    st.stop()

# ---------------- Prepare data (interactions, coercion) ----------------
data = df.copy()

# Build predictor DataFrame
X = data[preds].copy()

if add_interactions:
    for i in range(len(preds)):
        for j in range(i+1, len(preds)):
            a, b = preds[i], preds[j]
            name = f"{a}__x__{b}"
            # create interaction but avoid if non-numeric
            try:
                X[name] = pd.to_numeric(data[a], errors='coerce') * pd.to_numeric(data[b], errors='coerce')
            except Exception:
                X[name] = np.nan

selected_cols = [dep] + list(X.columns)

# Optionally drop rows with NaNs
if drop_na_rows:
    # before drop, coerce selected cols to numeric (so non-numeric become NaN and get dropped)
    data_sub = data[selected_cols].apply(pd.to_numeric, errors='coerce').dropna()
else:
    data_sub = data[selected_cols].apply(pd.to_numeric, errors='coerce')

if data_sub.shape[0] == 0:
    st.error("No valid rows after coercion/dropna. Check that selected columns are numeric or that drop_na is not removing everything.")
    st.stop()

# Ensure we have more observations than parameters
n_obs = data_sub.shape[0]
n_params = len(X.columns) + (1 if intercept else 0)
if n_obs <= n_params:
    st.error(f"Not enough observations for regression: {n_obs} rows vs {n_params} parameters (need >). Try removing predictors or disable intercept.")
    st.stop()

# Run when auto_run or manual button pressed
if (auto_run and uploaded_file) or manual_run:
    y = data_sub[dep].astype(float)
    X_final = data_sub[X.columns].astype(float)
    if intercept:
        X_final = sm.add_constant(X_final, has_constant='add')

    # Fit OLS
    try:
        model = sm.OLS(y, X_final)
        results = model.fit()
    except Exception as e:
        st.error(f"Regression failed: {e}")
        st.stop()

    # ------- Coefficient table -------
    params = results.params
    bse = results.bse
    tvals = results.tvalues
    pvals = results.pvalues
    conf = results.conf_int(alpha=0.05)
    conf.columns = ['CI_lower', 'CI_upper']

    coef_df = pd.DataFrame({
        'Variable': params.index,
        'Coef': params.values,
        'Std Err': bse.values,
        't': tvals.values,
        'p-value': pvals.values
    }).set_index('Variable')
    coef_df[['CI_lower','CI_upper']] = conf
    def signif_stars(p):
        if pd.isna(p): return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        if p < 0.1: return "."
        return ""
    coef_df['signif'] = coef_df['p-value'].apply(signif_stars)
    coef_show = coef_df.copy()
    coef_show[['Coef','Std Err','t','p-value','CI_lower','CI_upper']] = coef_show[[
        'Coef','Std Err','t','p-value','CI_lower','CI_upper']].round(6)

    # ------- Model stats (SSR, ESS, TSS, RMSE, F, Prob(F), AIC/BIC) -------
    ssr = float(results.ssr)
    try:
        tss = float(results.centered_tss)
    except Exception:
        tss = float(((y - y.mean())**2).sum())
    ess = float(results.ess) if hasattr(results, 'ess') else tss - ssr
    mse_resid = results.mse_resid
    rmse = np.sqrt(mse_resid)
    fstat = float(results.fvalue) if results.fvalue is not None else np.nan
    f_pval = float(results.f_pvalue) if results.f_pvalue is not None else np.nan

    model_stats = {
        'Observations': int(results.nobs),
        'Df Model': int(results.df_model),
        'Df Residuals': int(results.df_resid),
        'R-squared': results.rsquared,
        'Adj. R-squared': results.rsquared_adj,
        'F-statistic': fstat,
        'Prob (F-statistic)': f_pval,
        'SSR (resid sum of squares)': ssr,
        'ESS (explained sum squares)': ess,
        'TSS (total sum squares)': tss,
        'MSE (resid)': mse_resid,
        'RMSE (root mse)': rmse,
        'Log-Likelihood': results.llf,
        'AIC': results.aic,
        'BIC': results.bic
    }

    # ------- ANOVA (Type II preferred) -------
    try:
        anova = sm.stats.anova_lm(results, typ=2).reset_index().rename(columns={'index':'Variable'})
    except Exception:
        # fallback: simple summary with Model vs Residual
        anova = pd.DataFrame([
            {'Variable':'Model', 'df': results.df_model, 'sum_sq': ess, 'mean_sq': ess / results.df_model if results.df_model>0 else np.nan, 'F': results.fvalue, 'PR(>F)': results.f_pvalue},
            {'Variable':'Residual', 'df': results.df_resid, 'sum_sq': ssr, 'mean_sq': results.mse_resid}
        ])

    # ------- Predictions + residuals -------
    preds = results.predict(X_final)
    out_df = data_sub.copy()
    out_df['_y_true'] = y
    out_df['_y_pred'] = preds
    out_df['_residual'] = out_df['_y_true'] - out_df['_y_pred']

    # ------- Display UI -------
    left, right = st.columns([2,1])
    with left:
        st.subheader("Coefficients (Sheet1-style)")
        st.dataframe(coef_show.reset_index())
        st.markdown("**Significance:** `*** p<0.001`, `** p<0.01`, `* p<0.05`, `. p<0.1`")
        st.subheader("ANOVA")
        st.dataframe(anova)
    with right:
        st.subheader("Model statistics")
        stats_df = pd.DataFrame(list(model_stats.items()), columns=['Stat','Value'])
        def fmt(v):
            if isinstance(v, (int, np.integer)): return f"{v}"
            if isinstance(v, (float, np.floating)): return f"{v:.6g}"
            return str(v)
        stats_df['Value'] = stats_df['Value'].apply(fmt)
        st.table(stats_df)

    # Diagnostic plots
    st.subheader("Diagnostic plots")
    c1, c2 = st.columns(2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(preds, out_df['_residual'], alpha=0.6)
    ax1.axhline(0, linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    c1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    stats.probplot(out_df['_residual'], dist="norm", plot=ax2)
    ax2.set_title("QQ plot")
    c2.pyplot(fig2)

    # Downloads
    buf_coef = io.StringIO()
    coef_df.reset_index().to_csv(buf_coef, index=False)
    buf_coef.seek(0)
    st.download_button("Download coefficients (CSV)", data=buf_coef, file_name="coefficients.csv", mime="text/csv")

    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
        coef_df.reset_index().to_excel(writer, sheet_name='Coefficients', index=False)
        pd.DataFrame(list(model_stats.items()), columns=['Stat','Value']).to_excel(writer, sheet_name='ModelStats', index=False)
        anova.to_excel(writer, sheet_name='ANOVA', index=False)
        out_df.reset_index().to_excel(writer, sheet_name='Predictions', index=False)
    buf_xlsx.seek(0)
    st.download_button("Download full regression output (XLSX)", data=buf_xlsx, file_name="regression_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("Full statsmodels textual summary"):
        st.text(results.summary().as_text())
