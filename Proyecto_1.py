import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard de Riesgo Financiero", layout="wide")

# --- CARGA DE DATOS (Función optimizada) ---
@st.cache_data
def load_data(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=False)
    prices = df["Adj Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

# --- SIDEBAR ---
st.sidebar.header("Parámetros de Simulación")
stock = st.sidebar.text_input("Ticker", value="LMT")
start_date = st.sidebar.date_input("Fecha de Inicio", value=pd.to_datetime("2010-05-01"))
window = st.sidebar.slider("Ventana Móvil (Días)", 100, 500, 252)
n_sim = st.sidebar.number_input("Simulaciones Monte Carlo", value=10000)

# Obtener retornos
returns = load_data(stock, start_date)

# --- ENCABEZADO PRINCIPAL ---
st.title(f"📊 Análisis de Riesgo: {stock}")
st.markdown("Cálculo de VaR y ES mediante múltiples metodologías y Backtesting.")

# --- MÉTRICAS DESCRIPTIVAS ---
col1, col2, col3 = st.columns(3)
mu = returns.mean()
sigma = returns.std(ddof=1)
nu, loc, scale = t.fit(returns)

col1.metric("Media Diaria", f"{mu:.5%}")
col2.metric("Sesgo (Skewness)", f"{returns.skew():.4f}")
col3.metric("Curtosis", f"{returns.kurtosis():.4f}")

# --- SECCIÓN DE TEXTOS Y ANÁLISIS ---
with st.expander("📌 1. Introducción"):
    st.markdown("""
    En este proyecto estimamos el riesgo de mercado mediante dos métricas fundamentales: el **Value at Risk (VaR)** y el **Expected Shortfall (ES)**. 
    El objetivo es evaluar qué tan bien se adaptan distintos modelos matemáticos (paramétricos, históricos y simulaciones) a la realidad del mercado, utilizando ventanas móviles para evitar el sesgo de anticipación y finalizando con un Backtesting para comprobar su eficiencia real.
    """)

with st.expander(f"🎯 2. ¿Por qué elegí esta acción ({stock})?"):
    st.markdown(f"""
    **{stock}** es una empresa interesante para el análisis de riesgo por las siguientes razones:
    * **Sector Estratégico:** Al ser del sector defensa, reacciona fuertemente a eventos geopolíticos y presupuestos gubernamentales.
    * **Perfil de Riesgo:** Presenta una volatilidad particular comparada con el S&P500, lo que permite testear la robustez de las colas pesadas.
    * **Dividendos y Estabilidad:** A pesar de ser volátil ante noticias, tiene una tendencia de crecimiento a largo plazo que afecta la media de los retornos.
    """)

with st.expander("📊 3. ¿Qué podemos observar en los diferentes modelos?"):
    st.markdown("""
    Al comparar las gráficas y el backtesting, destacan los siguientes comportamientos:
    
    * **VaR Normal:** Suele subestimar el riesgo en activos financieros porque no captura el "fat tail" (eventos extremos).
    * **VaR t-Student:** Se adapta mejor a Lockheed Martin al permitir mayor probabilidad en los extremos de la distribución.
    * **VaR Volatilidad Móvil (Media cero):** Es más reactivo a picos de pánico recientes en el mercado, siendo más conservador cuando la volatilidad sube bruscamente.
    """)

with st.expander("🏁 4. Conclusión"):
    # Nota: El usuario puede llenar esto tras ver los resultados del Backtesting en la pestaña 3
    st.markdown("""
    Tras realizar el backtesting y contar el número de violaciones, podemos determinar qué modelo es más preciso. 
    Un modelo ideal debería tener un **% de violaciones cercano al 1.00%** para un nivel de confianza del 99%.
    """)

st.divider() 

# --- PESTAÑAS PARA ORGANIZAR EL CONTENIDO ---
tab1, tab2, tab3 = st.tabs(["📈 Gráficas de Retorno", "🛡️ Riesgo Móvil", "📉 Backtesting"])

with tab1:
    st.subheader("Distribución de Retornos")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.hist(returns, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.set_title(f"Histograma de Frecuencias - {stock}")
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
    df_risk['VaR_99_Vol'] = q_norm * roll_std 
    
    df_risk = df_risk.dropna()

    fig_risk, ax_risk = plt.subplots(figsize=(12, 6))
    ax_risk.plot(df_risk.index, df_risk['Return'], color='lightgrey', label='Retorno Diario', alpha=0.5)
    ax_risk.plot(df_risk.index, df_risk['VaR_99_Norm'], label='VaR 99% Normal', linestyle=':')
    ax_risk.plot(df_risk.index, df_risk['VaR_99_t'], label='VaR 99% t-Student', linestyle='--')
    ax_risk.plot(df_risk.index, df_risk['VaR_99_Vol'], label='VaR 99% Vol Móvil ($\mu=0$)', color='black')
    ax_risk.set_title("Comparación de Modelos VaR (99% Confianza)")
    ax_risk.legend()
    st.pyplot(fig_risk)

with tab3:
    st.subheader("Análisis de Violaciones (Backtesting)")
    
    res_backtest = []
    cols_to_test = ['VaR_99_Norm', 'VaR_99_t', 'VaR_99_Vol']
    
    for c in cols_to_test:
        viols = (df_risk['Return'] < df_risk[c]).sum()
        pct = (viols / len(df_risk)) * 100
        # Criterio de estado
        if 0.8 <= pct <= 1.5:
            estado = "✅ Óptimo"
        elif pct < 0.8:
            estado = "🔵 Conservador"
        else:
            estado = "⚠️ Riesgoso (Subestima)"
            
        res_backtest.append({
            "Modelo": c,
            "Total Violaciones": viols,
            "% Violaciones": f"{pct:.2f}%",
            "Estado": estado
        })
    
    st.table(pd.DataFrame(res_backtest))
    st.info("""
    **¿Cómo leer esta tabla?** Si el % de violaciones es mucho mayor al 1%, el modelo es 'peligroso' porque el mercado cayó con fuerza más veces de las que el modelo predijo. Si es mucho menor, el modelo es demasiado precavido.
    """)