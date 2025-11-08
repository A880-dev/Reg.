# main.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import io
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="MultiRegressor — Auto-run Sheet1-style", layout="wide")
st.title("MultiRegressor — Upload data and get Sheet1-style OLS (auto-run)")

st.markdown(
    "Upload CSV / Excel. The app will try to auto-detect numeric columns and run a full regression automatically. You can override variable choices in the sidebar."
)

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload CSV / XLS / XLSX", type=["csv", "xls", "xlsx"])
if not uploaded_file:
    st.info("Upload a file to begin.")
    st.stop()

# ---------- Read file and choose sheet ----------
file_name = uploaded_file.name.lower()
if file_name.endswith(".csv"):
    try:
        df = pd.read_csv(uploaded_file)
        sheet_names = ["CSV"]
        sheet_choice = "CSV"
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    try:
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        sheet_choice = st.selectbox("Select sheet (Excel)", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_choice)
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        st.stop()

st.subheader("Data preview")
st.dataframe(df.head(200))

# ---------- Auto-detect numeric columns and choose defaults ----------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# heuristics for Y column names commonly used
preferred_y_names = ['y', 'dependent', 'target', 'outcome', 'response']
auto_y = None
for name in preferred_y_names:
    for col in all_cols:
        if col.strip().lower() == name:
            auto_y = col
            break
    if auto_y:
        break

if not auto_y and numeric_cols:
    # choose first numeric as default Y
    auto_y = numeric_cols[0]

auto_xs = []
if numeric_cols:
    auto_xs = [c for c in numeric_cols if c != auto_y]

# ---------- Sidebar controls ----------
st.sidebar.header("Regression controls")
auto_run = st.sidebar.checkbox("Auto-run on upload / change (recommended)", value=True)
dep = st.sidebar.selectbox("Dependent (Y) — override", options=[""] + all_cols, index=(1 + all_cols.index(auto_y) if auto_y in all_cols else 0))
# If user selected empty string treat as auto
if dep == "":
    dep = auto_y

preds = st.sidebar.multiselect("Predictors (X) — override", options=[c for c in all_cols if c != dep], default=auto_xs)
intercept = st.sidebar.checkbox("Include intercept", value=True)
drop_na_rows = st.sidebar.checkbox("Drop rows with missing values in selected columns", value=True)
add_interactions = st.sidebar.checkbox("Add pairwise interactions (X1*X2)", value=False)
manual_run = st.sidebar.button("Run regression (manual)")

# Decide whether to run: manual button OR (auto_run True and file uploaded)
should_run = manual_run or auto_run

# If preds empty & we have numeric columns, fall back to auto_xs
if (not preds or len(preds) == 0) and auto_xs:
    preds = auto_xs

# Guard
if not dep or not preds:
    st.warning("Dependent or predictors not set. The app suggested defaults in the sidebar — change them or press Run.")
    if not should_run:
        st.stop()

# ---------- Prepare data ----------
data = df.copy()

# build X with optional interactions
X = data[preds].copy()
if add_interactions:
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i], preds[j]
            name = f"{a}__x__{b}"
            X[name] = data[a] * data[b]

selected_cols = [dep] + list(X.columns)
if drop_na_rows:
    data2 = data[selected_cols].dropna()
else:
    data2 = data[selected_cols]

# Try coercion to numeric for the columns involved
data2 = data2.apply(pd.to_numeric, errors='coerce')
if drop_na_rows:
    data2 = data2.dropna()

if data2.shape[0] <= len(X.columns):
    st.error("Not enough observations after processing (need more rows than parameters).")
    st.stop()

# Run when required
if should_run:
    y = data2[dep].astype(float)
    X_final = data2[X.columns].astype(float)
    if intercept:
        X_final = sm.add_constant(X_final, has_constant='add')

    try:
        model = sm.OLS(y, X_final)
        results = model.fit()
    except Exception as e:
        st.error(f"Regression failed: {e}")
        st.stop()

    # --- Coeff table ---
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

    # --- Model stats ---
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
        "Df Residuals": int(results.df_resid),
        "Df Model": int(results.df_model),
        "R-squared": results.rsquared,
        "Adj. R-squared": results.rsquared_adj,
        "F-statistic": fstat,
        "Prob (F-statistic)": f_pval,
        "SSR (resid sum of squares)": ssr,
        "ESS (explained sum sq)": ess,
        "TSS (total sum sq)": tss,
        "MSE (resid)": mse_resid,
        "RMSE (root mse)": rmse,
        "Log-Likelihood": results.llf,
        "AIC": results.aic,
        "BIC": results.bic
    }

    # --- ANOVA ---
    try:
        anova = sm.stats.anova_lm(results, typ=2).reset_index().rename(columns={'index':'Variable'})
    except Exception:
        anova = pd.DataFrame([{
            'Variable': 'Model',
            'df': results.df_model,
            'sum_sq': ess,
            'mean_sq': ess / results.df_model if results.df_model>0 else np.nan,
            'F': results.fvalue,
            'PR(>F)': results.f_pvalue
        }, {
            'Variable': 'Residual',
            'df': results.df_resid,
            'sum_sq': ssr,
            'mean_sq': results.mse_resid
        }])

    # --- Predictions & residuals ---
    preds = results.predict(X_final)
    out_df = data2.copy()
    out_df['_y_true'] = y
    out_df['_y_pred'] = preds
    out_df['_residual'] = out_df['_y_true'] - out_df['_y_pred']

    # --- Display ---
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
    col1, col2 = st.columns(2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(preds, out_df['_residual'], alpha=0.6)
    ax1.axhline(0, linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    stats.probplot(out_df['_residual'], dist="norm", plot=ax2)
    ax2.set_title("QQ plot of residuals")
    col2.pyplot(fig2)

    # Downloads
    buf_coef = io.StringIO()
    coef_df.reset_index().to_csv(buf_coef, index=False)
    buf_coef.seek(0)
    st.download_button("Download coefficients (CSV)", data=buf_coef, file_name="regression_coefficients.csv", mime="text/csv")

    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine='openpyxl') as writer:
        coef_df.reset_index().to_excel(writer, sheet_name='Coefficients', index=False)
        pd.DataFrame(list(model_stats.items()), columns=['Stat','Value']).to_excel(writer, sheet_name='ModelStats', index=False)
        anova.to_excel(writer, sheet_name='ANOVA', index=False)
        out_df.reset_index().to_excel(writer, sheet_name='Predictions', index=False)
    buf_xlsx.seek(0)
    st.download_button("Download full regression output (XLSX)", data=buf_xlsx, file_name="regression_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with st.expander("Full regression summary (text)"):
        st.text(results.summary().as_text())
else:
    st.info("Auto-run disabled; adjust choices and press 'Run regression (manual)' in the sidebar.")
