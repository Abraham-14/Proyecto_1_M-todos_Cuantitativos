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
    # Manejar MultiIndex si ocurre
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
En este proyecto estimamos los riesgos de mercado mediante dos medidas fundamentales: **Value at Risk (VaR)** y el **Expected Shortfall (ES)**. El objetivo es calcular qué tan bien se adaptan los distintos modelos en su forma paramétrica, histórica y simulada. Adicionalmente, utilizamos ventanas móviles (*rolling windows*) para evitar el sesgo de anticipación, finalizando con un *Backtesting* para comprobar su eficacia real en el mercado.
""")

with st.expander("¿Por qué Lockheed Martin (LMT)?", expanded=True):
    st.write("""
    Dado que el precio de las acciones refleja el entorno macroeconómico y geopolítico, un tema de alta relevancia actual son los conflictos bélicos (por ejemplo, las tensiones entre Israel e Irán, y la guerra entre Rusia y Ucrania). 
    
    En estos escenarios, el armamento militar es un factor clave. Es aquí donde entra **Lockheed Martin**, el contratista militar número uno a nivel mundial por ventas de defensa. Considerando que aproximadamente el 73% de sus ingresos provienen del gobierno de los Estados Unidos, resulta fascinante analizar matemáticamente qué tan grandes pueden ser sus pérdidas extremas (riesgo de cola) y su volatilidad ante las noticias de nuestra realidad actual.
    """)

# --- 2. ESTADÍSTICA DESCRIPTIVA (Inciso B) ---
st.header("1. Análisis Descriptivo de los Retornos")
media = returns.mean()
sesgo = returns.skew()
curtosis = returns.kurtosis() # Fisher (Exceso de curtosis)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Observaciones", len(returns))
c2.metric("Media Diaria", f"{media:.6f}")
c3.metric("Sesgo (Skewness)", f"{sesgo:.4f}")
c4.metric("Exceso de Curtosis", f"{curtosis:.4f}")

st.info(f"""
**Análisis de Distribución:** Antes de abordar los modelos, el análisis descriptivo revela un nivel de exceso de curtosis altísimo (**{curtosis:.2f}**). Esto nos indica contundentemente que la distribución de LMT presenta **"colas pesadas"** (leptocurtosis). Por lo tanto, asumir una distribución Normal tradicional sería un error; la distribución **t-Student** se perfila como nuestra mejor candidata teórica. De hecho, al realizar pruebas gráficas (Q-Q plot) y el test de Kolmogorov-Smirnov (K-S) en nuestro análisis previo, confirmamos que los rendimientos se ajustan a una t-Student.
""")

# --- 3. VaR y ES ESTÁTICO (Inciso C) ---
st.header("2. Estimación Estática: VaR y ES (Serie Completa)")

@st.cache_data
def calc_static_risk(rets, n_sim):
    niveles = [0.95, 0.975, 0.99]
    res = []
    mu, sigma = rets.mean(), rets.std(ddof=1)
    
    # Ajuste t-Student
    df_t, loc_t, scale_t = t.fit(rets)
    # Monte Carlo Simulación
    np.random.seed(42)
    sim_mc = t.rvs(df_t, loc=loc_t, scale=scale_t, size=n_sim)
    
    for conf in niveles:
        alpha = 1 - conf
        # Histórico
        vh = np.percentile(rets, alpha*100)
        esh = rets[rets <= vh].mean()
        # Normal
        z = norm.ppf(alpha)
        vn = mu + z * sigma
        esn = mu - sigma * (norm.pdf(z)/alpha)
        # t-Student
        ts = t.ppf(alpha, df_t)
        vt = loc_t + ts * scale_t
        est = loc_t - scale_t * (t.pdf(ts, df_t)/alpha) * ((df_t + ts**2)/(df_t - 1))
        # Monte Carlo
        vmc = np.percentile(sim_mc, alpha*100)
        esmc = sim_mc[sim_mc <= vmc].mean()
        
        res.append([f"{conf*100}%", vh, esh, vn, esn, vt, est, vmc, esmc])
    
    return pd.DataFrame(res, columns=['Confianza', 'VaR Hist', 'ES Hist', 'VaR Norm', 'ES Norm', 'VaR t-Stud', 'ES t-Stud', 'VaR MC', 'ES MC'])

df_estatico = calc_static_risk(returns, n_sim)
st.dataframe(df_estatico.style.format({c: "{:.4f}" for c in df_estatico.columns if c != 'Confianza'}), use_container_width=True)

st.success("""
**Conclusiones de la Tabla Estática:**
* **VaR t-Student:** Notamos que con un nivel de confianza del 95% (1 de cada 20 días) la pérdida esperada promedio rondaría el -1.86%. Al 99% de confianza (1 de cada 100 días, es decir 2 o 3 veces al año), el golpe es mucho más severo (-3.56%).
* **ES Histórico y Normalidad:** Al ver los diferentes modelos aplicados en el ES, notamos que asumir normalidad arroja estimaciones menos severas. Sin embargo, al ver el histórico y la t-Student, estos nos dan una pérdida esperada mucho mayor. Esto reafirma que nuestro modelo es de colas pesadas y consolida a la t-Student como nuestra mejor candidata.
""")

# --- 4. ROLLING WINDOW (Incisos D y F) ---
st.header(f"3. Análisis Dinámico: Ventanas Móviles ({window} días)")
st.markdown("Cálculo del riesgo para el día $t+1$ utilizando estrictamente la información histórica disponible hasta el día $t$.")

n = len(returns)
# Inicializar arrays con NaNs
var_h_95, var_h_99 = np.full(n, np.nan), np.full(n, np.nan)
es_h_95, es_h_99 = np.full(n, np.nan), np.full(n, np.nan)
var_n_95, var_n_99 = np.full(n, np.nan), np.full(n, np.nan)
es_n_95, es_n_99 = np.full(n, np.nan), np.full(n, np.nan)
var_vol_95, var_vol_99 = np.full(n, np.nan), np.full(n, np.nan)

z_95, z_99 = norm.ppf(0.05), norm.ppf(0.01)
ret_vals = returns.values

# Lógica del Rolling Window (Predecir t+1 sin sesgo de anticipación)
for i in range(window, n):
    win = ret_vals[i - window : i] # Solo ve el pasado
    
    # Histórico
    vh95, vh99 = np.percentile(win, 5), np.percentile(win, 1)
    var_h_95[i], var_h_99[i] = vh95, vh99
    es_h_95[i], es_h_99[i] = win[win <= vh95].mean(), win[win <= vh99].mean()
    
    # Normal Paramétrico
    mu, sigma = np.mean(win), np.std(win, ddof=1)
    var_n_95[i], var_n_99[i] = mu + z_95*sigma, mu + z_99*sigma
    es_n_95[i] = mu - sigma * (norm.pdf(z_95)/0.05)
    es_n_99[i] = mu - sigma * (norm.pdf(z_99)/0.01)
    
    # Inciso F: VaR de Volatilidad Móvil (sigma_t * q_alpha asumiendo mu=0)
    var_vol_95[i], var_vol_99[i] = z_95 * sigma, z_99 * sigma

# Consolidar en DataFrame
df_dyn = pd.DataFrame(index=returns.index)
df_dyn['Retornos'] = returns
df_dyn['VaR_H_99'], df_dyn['ES_H_99'] = var_h_99, es_h_99
df_dyn['VaR_N_99'], df_dyn['ES_N_99'] = var_n_99, es_n_99
df_dyn['VaR_Vol_99'] = var_vol_99

# Gráfica Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['Retornos'], mode='lines', name='Retornos (P&L)', line=dict(color='lightgray', width=1), opacity=0.7))
fig.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['VaR_N_99'], mode='lines', name='VaR 99% (Normal)', line=dict(color='blue', dash='dot')))
fig.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['VaR_H_99'], mode='lines', name='VaR 99% (Histórico)', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['ES_H_99'], mode='lines', name='ES 99% (Histórico)', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df_dyn.index, y=df_dyn['VaR_Vol_99'], mode='lines', name='VaR 99% Vol Móvil (μ=0)', line=dict(color='black', width=1)))

fig.update_layout(title='Evolución del Riesgo al 99% (Rolling Window)', xaxis_title='Fecha', yaxis_title='Rendimientos', template='plotly_white', hovermode='x unified', height=600)
st.plotly_chart(fig, use_container_width=True)

st.warning("""
### Análisis Visual de las Gráficas y la Ventana Móvil:
De la gráfica interactiva anterior, y de los análisis previos generados, podemos concluir:

1. **Adaptabilidad a la realidad:** La gráfica demuestra que la ventana móvil es necesaria porque ajusta el nivel de "alarma" a la realidad actual. Aumenta el riesgo en las crisis y lo disminuye en tiempos de paz. Por ejemplo, cuando sucede la crisis del COVID-19 en 2020, existe una caída preocupante y las líneas del VaR/ES caen a un pozo para proteger la cartera.
2. **El "Efecto Memoria" del Histórico:** Visualmente, el método histórico se dibuja como "escalones" o líneas planas durante las crisis. Esto ocurre porque la ventana se "trauma" con la caída severa y recuerda ese peor día durante exactamente 252 días, hasta que el dato caduca y la línea vuelve a estabilizarse.
3. **El VaR no es suficiente (La importancia del ES):** Visualmente comprobamos cómo los picos grises (los retornos) logran perforar y caer por debajo de las líneas de VaR en momentos de estrés extremo. El VaR solo nos marca la frontera, pero necesitamos el apoyo del ES (Expected Shortfall) para capturar toda la profundidad de esas pérdidas fuera del límite.
4. **Validación del Supuesto $\mu=0$:** La línea negra (VaR Volatilidad Móvil) y la línea punteada (VaR Normal Clásico) son prácticamente un clon visual. Esto demuestra que asumir la media como cero en retornos diarios de alta frecuencia es un atajo válido que no sacrifica precisión.
""")

# --- 5. BACKTESTING Y VIOLACIONES (Incisos E y F) ---
st.header("4. Backtesting: Eficiencia de los Modelos")
st.markdown("Un modelo eficiente al 99% de confianza debería tener un porcentaje de violaciones cercano al 1% (y siempre menor al 2.5% según el límite de tolerancia aceptable).")

df_test = df_dyn.dropna()
n_obs = len(df_test)

def count_violations(col_risk):
    return (df_test['Retornos'] < df_test[col_risk]).sum()

viol_data = {
    'Métrica de Riesgo': ['VaR 99% Normal', 'ES 99% Normal', 'VaR 99% Histórico', 'ES 99% Histórico', 'VaR 99% Vol. Móvil (Inciso F)'],
    'Violaciones (#)': [count_violations('VaR_N_99'), count_violations('ES_N_99'), count_violations('VaR_H_99'), count_violations('ES_H_99'), count_violations('VaR_Vol_99')],
}

df_viol = pd.DataFrame(viol_data)
df_viol['% de la Muestra'] = (df_viol['Violaciones (#)'] / n_obs) * 100

st.table(df_viol.style.format({'% de la Muestra': "{:.2f}%"}))

st.success("""
**Evaluación Final:** El backtesting revela claramente la debilidad de los modelos paramétricos asumiendo normalidad frente a activos con colas pesadas. En este caso particular, notamos que para la distribución t-Student / Histórica se asumen mayores pérdidas, ajustándose de manera más conservadora a la realidad de las crisis bélicas y de salud observadas en la historia de LMT.
""")
