import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, t
import scipy.stats as stats

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Proyecto Riesgo MCF", layout="wide", page_icon="📊")

# --- 1. CARGA DE DATOS (Inciso A) ---
@st.cache_data
def load_data(ticker, start_date):
    df = yf.download(ticker, start=start_date, auto_adjust=False)
    prices = df['Adj Close'][ticker] if isinstance(df.columns, pd.MultiIndex) else df['Adj Close']
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns

# --- SIDEBAR PRINCIPAL ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Lockheed_Martin_logo.svg/1200px-Lockheed_Martin_logo.svg.png", width=200)
st.sidebar.header("Parámetros del Proyecto")
stock = st.sidebar.text_input("Activo (Ticker)", value="LMT")
start_date = st.sidebar.date_input("Fecha de Inicio", value=pd.to_datetime("2010-01-01"))
window = st.sidebar.slider("Ventana Móvil (Días)", 100, 500, 252)
n_sim = st.sidebar.number_input("Simulaciones Monte Carlo", value=10000)

prices, returns = load_data(stock, start_date)

# --- INTRODUCCIÓN Y JUSTIFICACIÓN ---
st.title(f"📊 Proyecto de Riesgo de Mercado: {stock}")

st.markdown("""
### Introducción
En este proyecto estimamos los riesgos de mercado mediante dos medidas fundamentales: **Value at Risk (VaR)** y el **Expected Shortfall (ES)**. El objetivo es calcular qué tan bien se adaptan los distintos modelos en su forma paramétrica, histórica y simulada.
""")

with st.expander("¿Por qué elegimos Lockheed Martin (LMT)?", expanded=True):
    st.write(f"Analizamos **{stock}** por su relevancia en el contexto geopolítico actual. Como principal contratista militar global, su volatilidad refleja las tensiones internacionales actuales, siendo un caso de estudio ideal para riesgos extremos.")

# --- 2. ESTADÍSTICA DESCRIPTIVA (Inciso B) ---
st.header("1. Análisis Descriptivo de los Retornos")
curtosis = returns.kurtosis()
st.info(f"**Análisis:** El exceso de curtosis de **{curtosis:.2f}** confirma colas pesadas. La distribución **t-Student** es nuestra mejor candidata teórica frente a la Normal.")

# --- 3. VaR y ES ESTÁTICO (Inciso C) + NUEVA GRÁFICA ---
st.header("2. Estimación Estática: VaR y ES (Serie Completa)")

@st.cache_data
def calc_static_risk(rets, n_sim):
    niveles = [0.95, 0.975, 0.99]
    res = []
    mu, sigma = rets.mean(), rets.std(ddof=1)
    df_t, loc_t, scale_t = t.fit(rets)
    np.random.seed(42)
    sim_mc = np.random.normal(mu, sigma, n_sim)
    
    for conf in niveles:
        alpha = 1 - conf
        vh = np.percentile(rets, alpha*100)
        esh = rets[rets <= vh].mean()
        z = norm.ppf(alpha)
        vn = mu + z * sigma
        esn = mu - sigma * (norm.pdf(z)/alpha)
        ts = t.ppf(alpha, df_t)
        vt = loc_t + ts * scale_t
        est = loc_t - scale_t * (t.pdf(ts, df_t)/alpha) * ((df_t + ts**2)/(df_t - 1))
        vmc = np.percentile(sim_mc, alpha*100)
        esmc = sim_mc[sim_mc <= vmc].mean()
        res.append([f"{conf*100}%", vh, esh, vn, esn, vt, est, vmc, esmc])
    
    return pd.DataFrame(res, columns=['Confianza', 'VaR Hist', 'ES Hist', 'VaR Norm', 'ES Norm', 'VaR t-Stud', 'ES t-Stud', 'VaR MC', 'ES MC'])

df_estatico = calc_static_risk(returns, n_sim)
st.dataframe(df_estatico.style.format({c: "{:.4f}" for c in df_estatico.columns if c != 'Confianza'}), use_container_width=True)

# --- NUEVA GRÁFICA COMPARATIVA ESTÁTICA ---
row_99 = df_estatico[df_estatico['Confianza'] == '99.0%'].iloc[0]
fig_comp = go.Figure()
methods = ['Histórico', 'Normal', 't-Student', 'Monte Carlo']
vars_99 = [row_99['VaR Hist'], row_99['VaR Norm'], row_99['VaR t-Stud'], row_99['VaR MC']]
ess_99 = [row_99['ES Hist'], row_99['ES Norm'], row_99['ES t-Stud'], row_99['ES MC']]

fig_comp.add_trace(go.Bar(name='VaR 99%', x=methods, y=vars_99, marker_color='rgb(55, 83, 109)'))
fig_comp.add_trace(go.Bar(name='ES 99%', x=methods, y=ess_99, marker_color='rgb(26, 118, 255)'))
fig_comp.update_layout(title='Comparativa de Modelos Estáticos (Nivel 99%)', barmode='group', template='plotly_white', yaxis_title='Rendimiento')
st.plotly_chart(fig_comp, use_container_width=True)

# --- 4. ROLLING WINDOWS (Incisos D, E, F) ---
st.header(f"3. Análisis Dinámico: Ventanas Móviles ({window} días)")

n = len(returns)
var_h_99, es_h_99 = np.full(n, np.nan), np.full(n, np.nan)
var_n_99, es_n_99 = np.full(n, np.nan), np.full(n, np.nan)
var_t_99, es_t_99 = np.full(n, np.nan), np.full(n, np.nan) # NUEVO: t-Student dinámica
var_vol_99 = np.full(n, np.nan)

z_99 = norm.ppf(0.01)
ret_vals = returns.values

# Barra de progreso para el cálculo que es más intensivo con t.fit
bar = st.progress(0)
for i in range(window, n):
    win = ret_vals[i - window : i]
    
    # Histórico
    vh99 = np.percentile(win, 1)
    var_h_99[i], es_h_99[i] = vh99, win[win <= vh99].mean()
    
    # Normal
    mu, sigma = np.mean(win), np.std(win, ddof=1)
    var_n_99[i] = mu + z_99 * sigma
    es_n_99[i] = mu - sigma * (norm.pdf(z_99)/0.01)
    
    # t-Student Dinámica (Ajuste por ventana)
    dft, loct, scalet = t.fit(win)
    t99 = t.ppf(0.01, dft)
    var_t_99[i] = loct + t99 * scalet
    es_t_99[i] = loct - scalet * (t.pdf(t99, dft)/0.01) * ((dft + t99**2)/(dft - 1))
    
    # Inciso F
    var_vol_99[i] = z_99 * sigma
    
    if i % 100 == 0: bar.progress(i/n)
bar.empty()

df_dyn = pd.DataFrame(index=returns.index)
df_dyn['Retornos'] = returns
df_dyn['VaR_H_99'], df_dyn['ES_H_99'] = var_h_99, es_h_99
df_dyn['VaR_N_99'], df_dyn['ES_N_99'] = var_n_99, es_n_99
df_dyn['VaR_T_99'], df_dyn['ES_T_99'] = var_t_99, es_t_99
df_dyn['VaR_Vol_99'] = var_vol_99

# Gráfica Dinámica
fig_dyn = go.Figure()
fig_dyn.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['Retornos'], name='Retornos', line=dict(color='lightgray', width=1), opacity=0.5))
fig_dyn.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['VaR_H_99'], name='VaR Histórico', line=dict(color='orange')))
fig_dyn.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['VaR_T_99'], name='VaR t-Student', line=dict(color='green', dash='dash')))
fig_dyn.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['ES_T_99'], name='ES t-Student', line=dict(color='darkgreen')))
fig_dyn.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['VaR_Vol_99'], name='VaR Vol. Móvil', line=dict(color='black', width=1)))

fig_dyn.update_layout(title='Riesgo Dinámico al 99% (Rolling Window)', template='plotly_white', height=600)
st.plotly_chart(fig_dyn, use_container_width=True)

# --- 5. BACKTESTING FINAL (Con t-Student) ---
st.header("4. Backtesting: Eficiencia de los Modelos")
df_test = df_dyn.dropna()
n_obs = len(df_test)

def count_viol(col): return (df_test['Retornos'] < df_test[col]).sum()

viol_data = {
    'Modelo (99% Confianza)': ['Normal', 'Histórico', 't-Student (Dinámico)', 'Volatilidad Móvil (μ=0)'],
    'Violaciones VaR (#)': [count_viol('VaR_N_99'), count_viol('VaR_H_99'), count_viol('VaR_T_99'), count_viol('VaR_Vol_99')],
    'Violaciones ES (#)': [count_viol('ES_N_99'), count_viol('ES_H_99'), count_viol('ES_T_99'), "-"],
}

df_viol = pd.DataFrame(viol_data)
df_viol['% Violaciones VaR'] = (df_viol['Violaciones VaR (#)'] / n_obs) * 100

st.table(df_viol.style.format({'% Violaciones VaR': "{:.2f}%"}))

st.success("""
**Conclusión Final:** Al integrar la **t-Student**, observamos que es el modelo que mejor equilibra la realidad de las colas pesadas sin ser tan extremo como el histórico en periodos de post-crisis. El Expected Shortfall (ES) de la t-Student demuestra ser la métrica más robusta, capturando de manera eficiente las pérdidas que el VaR tradicional ignora.
""")