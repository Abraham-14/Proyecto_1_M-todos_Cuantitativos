import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard de Riesgo Financiero", layout="wide")
st.title("📊 Análisis de Riesgo: LMT (Lockheed Martin)")
st.markdown("Cálculo de VaR y ES mediante múltiples metodologías y Backtesting.")

# --- SIDEBAR (Entradas del usuario) ---
st.sidebar.header("Parámetros de Simulación")
stock = st.sidebar.text_input("Ticker", value="LMT")
start_date = st.sidebar.date_input("Fecha de Inicio", value=pd.to_datetime("2010-05-01"))
window = st.sidebar.slider("Ventana Móvil (Días)", 100, 500, 252)
n_sim = st.sidebar.number_input("Simulaciones Monte Carlo", value=10000)

# --- CARGA DE DATOS ---
@st.cache_data # Para que no descargue los datos cada vez que muevas un slider
def load_data(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=False)
    # Limpieza para nuevas versiones de yfinance
    prices = df["Adj Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

returns = load_data(stock, start_date)

# --- MÉTRICAS DESCRIPTIVAS ---
col1, col2, col3 = st.columns(3)
mu = returns.mean()
sigma = returns.std(ddof=1)
nu, loc, scale = t.fit(returns)

col1.metric("Media Diaria", f"{mu:.5%}")
col2.metric("Sesgo (Skewness)", f"{returns.skew():.4f}")
col3.metric("Curtosis", f"{returns.kurtosis():.4f}")

# --- PESTAÑAS PARA ORGANIZAR EL CONTENIDO ---
tab1, tab2, tab3 = st.tabs(["📈 Gráficas de Retorno", "🛡️ Riesgo Móvil", "📉 Backtesting"])

with tab1:
    st.subheader("Distribución de Retornos")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.hist(returns, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.set_title("Histograma de Frecuencias")
    st.pyplot(fig_hist)

    st.subheader("Rendimientos Mensuales")
    monthly = returns.resample("ME").sum()
    st.line_chart(monthly)

with tab2:
    st.subheader("Evolución del Riesgo (Rolling Windows)")
    
    # Cálculos Rolling
    df_risk = pd.DataFrame({'Return': returns})
    roll_mean = df_risk['Return'].rolling(window=window).mean().shift(1)
    roll_std = df_risk['Return'].rolling(window=window).std(ddof=1).shift(1)
    
    alpha_99 = 0.01
    q_norm = norm.ppf(alpha_99)
    q_t = t.ppf(alpha_99, df=nu)
    adj_scale = np.sqrt((nu - 2) / nu)

    # Métodos
    df_risk['VaR_99_Norm'] = roll_mean + (roll_std * q_norm)
    df_risk['VaR_99_t'] = roll_mean + (roll_std * adj_scale) * q_t
    df_risk['VaR_99_Vol'] = q_norm * roll_std # Tu última fórmula (mu=0)
    
    df_risk = df_risk.dropna()

    fig_risk, ax_risk = plt.subplots(figsize=(12, 6))
    ax_risk.plot(df_risk.index, df_risk['Return'], color='lightgrey', label='P&L Diario', alpha=0.5)
    ax_risk.plot(df_risk.index, df_risk['VaR_99_Norm'], label='VaR 99% Normal', linestyle=':')
    ax_risk.plot(df_risk.index, df_risk['VaR_99_t'], label='VaR 99% t-Student', linestyle='--')
    ax_risk.plot(df_risk.index, df_risk['VaR_99_Vol'], label='VaR 99% Vol Móvil ($\mu=0$)', color='black')
    ax_risk.legend()
    st.pyplot(fig_risk)

with tab3:
    st.subheader("Análisis de Violaciones (Eficiencia)")
    
    res_backtest = []
    cols_to_test = ['VaR_99_Norm', 'VaR_99_t', 'VaR_99_Vol']
    
    for c in cols_to_test:
        viols = (df_risk['Return'] < df_risk[c]).sum()
        pct = (viols / len(df_risk)) * 100
        res_backtest.append({
            "Modelo": c,
            "Total Violaciones": viols,
            "% Violaciones": f"{pct:.2f}%",
            "Estado": "✅ Óptimo" if pct <= 1.5 else "⚠️ Revisar"
        })
    
    st.table(pd.DataFrame(res_backtest))
    st.info("Nota: Para un VaR al 99%, se espera teóricamente un 1% de violaciones.")