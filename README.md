# Dashboard de Análisis de Riesgo Financiero (VaR y ES)

## 📌 Descripción del Proyecto
Este proyecto es una aplicación web interactiva desarrollada en **Python** utilizando **Streamlit**. Su objetivo es estimar y visualizar el riesgo de un activo financiero (por defecto, Lockheed Martin `LMT`) aplicando metodologías de **Value at Risk (VaR)** y **Expected Shortfall (ES)**.

El análisis incorpora el uso de **ventanas móviles (rolling windows)** de 252 días, garantizando que las predicciones del riesgo para el día $t+1$ utilicen únicamente información hasta el día $t$, evitando así el sesgo de anticipación (look-ahead bias).

### Metodologías Implementadas:
- VaR y ES Paramétrico (Distribución Normal).
- VaR y ES Paramétrico (Distribución t-Student ajustada por curtosis).
- VaR mediante Volatilidad Móvil (asumiendo $\mu = 0$).
- Backtesting: Cálculo de violaciones para medir la eficiencia de cada modelo.

---

## ⚙️ Requisitos previos (Instalación)
Para ejecutar este código, necesitas tener Python instalado y las siguientes librerías. Puedes instalarlas ejecutando este comando en tu terminal:

```bash
pip install streamlit yfinance pandas numpy matplotlib scipy