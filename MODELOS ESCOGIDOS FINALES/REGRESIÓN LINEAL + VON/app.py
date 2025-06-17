import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción Acuapónica - Modelo Combinado",
    page_icon="🐟🥬",
    layout="wide"
)

st.title("🐟🥬 Sistema de Predicción Acuapónica")
st.markdown("### Modelo Combinado: Von Bertalanffy + Regresión Lineal Multivariante")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Modelo")

# Funciones del modelo
def von_bertalanffy_combined(t, L_inf, t0, *beta_params):
    """
    Modelo combinado: Von Bertalanffy con k como función lineal de variables ambientales
    
    L(t) = L∞ · (1 - e^(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)(t-t₀)))
    
    Parámetros:
    - t: tiempo (días)
    - L_inf: longitud máxima teórica
    - t0: tiempo teórico inicial
    - beta_params: coeficientes de regresión [β₀, β₁, β₂, ..., βₙ]
    """
    # Extraer variables ambientales (pasadas como argumentos adicionales)
    n_betas = len(beta_params)
    beta0 = beta_params[0]
    
    # Para un solo punto en el tiempo (caso escalar)
    if np.isscalar(t):
        # Calcular k como combinación lineal de variables ambientales
        k = beta0
        return L_inf * (1 - np.exp(-k * (t - t0)))
    
    # Para múltiples puntos en el tiempo (caso array)
    result = np.zeros_like(t, dtype=float)
    for i in range(len(t)):
        # Calcular k como combinación lineal de variables ambientales
        k = beta0
        result[i] = L_inf * (1 - np.exp(-k * (t[i] - t0)))
    
    return result

def von_bertalanffy_env(t, env_data, L_inf, t0, beta_params):
    """
    Modelo combinado: Von Bertalanffy con k como función lineal de variables ambientales
    
    L(t) = L∞ · (1 - e^(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)(t-t₀)))
    
    Parámetros:
    - t: tiempo (días)
    - env_data: matriz de variables ambientales [X₁, X₂, ..., Xₙ]
    - L_inf: longitud máxima teórica
    - t0: tiempo teórico inicial
    - beta_params: coeficientes de regresión [β₀, β₁, β₂, ..., βₙ]
    """
    # Calcular k para cada punto en el tiempo
    k_values = beta_params[0] + np.sum(env_data * beta_params[1:], axis=1)
    
    # Aplicar Von Bertalanffy con k variable
    result = L_inf * (1 - np.exp(-k_values * (t - t0)))
    
    return result

def fit_combined_model(t_data, y_data, env_data, max_value_estimate=None):
    """
    Ajustar el modelo combinado Von Bertalanffy + Regresión Lineal
    
    Parámetros:
    - t_data: array de tiempos
    - y_data: array de valores observados (longitud, altura, etc.)
    - env_data: matriz de variables ambientales
    - max_value_estimate: estimación inicial de L_inf
    
    Retorna:
    - Diccionario con parámetros ajustados y predicciones
    """
    if max_value_estimate is None:
        max_value_estimate = np.max(y_data) * 1.2
    
    # Normalizar variables ambientales
    scaler = StandardScaler()
    env_scaled = scaler.fit_transform(env_data)
    
    # Número de variables ambientales
    n_vars = env_data.shape[1]
    
    # Función objetivo para optimización
    def objective_function(params):
        L_inf = params[0]
        t0 = params[1]
        betas = params[2:]
        
        # Calcular k para cada punto en el tiempo
        k_values = betas[0] + np.sum(env_scaled * betas[1:].reshape(-1, n_vars), axis=1)
        
        # Aplicar Von Bertalanffy
        y_pred = L_inf * (1 - np.exp(-k_values * (t_data - t0)))
        
        # Error cuadrático medio
        mse = np.mean((y_data - y_pred) ** 2)
        return mse
    
    # Parámetros iniciales [L_inf, t0, β₀, β₁, β₂, ..., βₙ]
    initial_params = [max_value_estimate, 0] + [0.05] + [0.01] * n_vars
    
    # Límites para los parámetros
    bounds = [
        (max_value_estimate * 0.5, max_value_estimate * 2.0),  # L_inf
        (-10, 10),  # t0
        (0.001, 0.5),  # β₀ (intercepto)
    ] + [(-0.2, 0.2) for _ in range(n_vars)]  # β₁, β₂, ..., βₙ
    
    # Optimización
    try:
        result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            # Extraer parámetros optimizados
            L_inf_opt = result.x[0]
            t0_opt = result.x[1]
            beta_params = result.x[2:]
            
            # Calcular k para cada punto en el tiempo
            k_values = beta_params[0] + np.sum(env_scaled * beta_params[1:].reshape(-1, n_vars), axis=1)
            
            # Calcular predicciones
            y_pred = L_inf_opt * (1 - np.exp(-k_values * (t_data - t0_opt)))
            
            return {
                'L_inf': L_inf_opt,
                't0': t0_opt,
                'beta_params': beta_params,
                'k_values': k_values,
                'y_pred': y_pred,
                'scaler': scaler,
                'success': True
            }
        else:
            raise Exception("Optimización no convergió")
    
    except Exception as e:
        st.warning(f"Error en optimización: {e}. Usando método alternativo.")
        
        # Método alternativo: Von Bertalanffy simple
        try:
            def von_bertalanffy_simple(t, L_inf, k, t0):
                return L_inf * (1 - np.exp(-k * (t - t0)))
            
            popt, _ = curve_fit(von_bertalanffy_simple, t_data, y_data, 
                              p0=[max_value_estimate, 0.05, 0], maxfev=5000)
            
            L_inf_opt, k_opt, t0_opt = popt
            y_pred = von_bertalanffy_simple(t_data, *popt)
            
            # Crear betas ficticios (solo intercepto)
            beta_params = np.zeros(n_vars + 1)
            beta_params[0] = k_opt
            
            return {
                'L_inf': L_inf_opt,
                't0': t0_opt,
                'beta_params': beta_params,
                'k_values': np.full_like(t_data, k_opt, dtype=float),
                'y_pred': y_pred,
                'scaler': scaler,
                'success': False
            }
        except:
            # Si todo falla, usar regresión lineal simple
            reg = LinearRegression()
            reg.fit(np.array(t_data).reshape(-1, 1), y_data)
            y_pred = reg.predict(np.array(t_data).reshape(-1, 1))
            
            return {
                'L_inf': max_value_estimate,
                't0': 0,
                'beta_params': np.zeros(n_vars + 1),
                'k_values': np.full_like(t_data, 0.05, dtype=float),
                'y_pred': y_pred,
                'scaler': scaler,
                'success': False
            }

def calculate_metrics(y_true, y_pred):
    """Calcular métricas de evaluación"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'R²': r2,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }

def evaluate_r2(r2):
    """Evaluar calidad del ajuste basado en R²"""
    if r2 >= 0.90:
        return "Excelente ajuste", "🟢"
    elif r2 >= 0.75:
        return "Bueno", "🟡"
    elif r2 >= 0.50:
        return "Aceptable", "🟠"
    else:
        return "Pobre ajuste", "🔴"

def predict_future(model_result, t_future, recent_env):
    """
    Predecir valores futuros usando el modelo combinado
    
    Parámetros:
    - model_result: resultado del ajuste del modelo
    - t_future: array de tiempos futuros
    - recent_env: valores ambientales recientes
    
    Retorna:
    - Array de predicciones
    """
    # Escalar variables ambientales
    env_scaled = model_result['scaler'].transform(recent_env)
    
    # Calcular k para cada punto futuro
    beta_params = model_result['beta_params']
    n_vars = env_scaled.shape[1]
    k_future = beta_params[0] + np.sum(env_scaled * beta_params[1:].reshape(-1, n_vars), axis=1)
    
    # Aplicar Von Bertalanffy
    L_inf = model_result['L_inf']
    t0 = model_result['t0']
    
    y_future = np.zeros_like(t_future, dtype=float)
    for i in range(len(t_future)):
        y_future[i] = L_inf * (1 - np.exp(-k_future[0] * (t_future[i] - t0)))
    
    return y_future

# Carga de archivos
st.sidebar.subheader("📁 Cargar Datos")

uploaded_truchas = st.sidebar.file_uploader(
    "Cargar datos de truchas (.xlsx)", 
    type=['xlsx'], 
    key="truchas"
)

uploaded_lechugas = st.sidebar.file_uploader(
    "Cargar datos de lechugas (.xlsx)", 
    type=['xlsx'], 
    key="lechugas"
)

# Botón para usar datos de ejemplo
if st.sidebar.button("🔄 Usar Datos de Ejemplo"):
    st.sidebar.success("Usando datos de ejemplo generados")

# Cargar datos
try:
    if uploaded_truchas is not None:
        df_truchas = pd.read_excel(uploaded_truchas)
    else:
        # Usar datos de ejemplo
        df_truchas = pd.read_excel('datos_truchas_arcoiris_acuaponia_10.xlsx')
    
    if uploaded_lechugas is not None:
        df_lechugas = pd.read_excel(uploaded_lechugas)
    else:
        # Usar datos de ejemplo
        df_lechugas = pd.read_excel('simulacion_lechuga_realista.xlsx')
    
    st.sidebar.success(f"✅ Truchas: {len(df_truchas)} registros")
    st.sidebar.success(f"✅ Lechugas: {len(df_lechugas)} registros")
    
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Configuración de predicción
st.sidebar.subheader("🔮 Configuración de Predicción")
dias_prediccion = st.sidebar.slider(
    "Días para predicción", 
    min_value=1, 
    max_value=365, 
    value=30, 
    step=1
)

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis Exploratorio", "🐟 Modelo Truchas", "🥬 Modelo Lechugas", "📈 Comparación y Métricas"])

with tab1:
    st.header("📊 Análisis Exploratorio de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐟 Datos de Truchas")
        st.dataframe(df_truchas.head(10))
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Descriptivas - Truchas")
        st.dataframe(df_truchas.describe())
        
    with col2:
        st.subheader("🥬 Datos de Lechugas")
        st.dataframe(df_lechugas.head(10))
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Descriptivas - Lechugas")
        st.dataframe(df_lechugas.describe())
    
    # Gráficos exploratorios
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Crecimiento Truchas', 'Variables Ambientales Truchas', 
                       'Crecimiento Lechugas', 'Variables Ambientales Lechugas'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Truchas - Crecimiento
    fig.add_trace(
        go.Scatter(x=df_truchas['Tiempo_dias'], y=df_truchas['Longitud_cm'],
                  mode='lines+markers', name='Longitud Truchas', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Truchas - Variables ambientales
    fig.add_trace(
        go.Scatter(x=df_truchas['Tiempo_dias'], y=df_truchas['Temperatura_C'],
                  mode='lines', name='Temperatura', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_truchas['Tiempo_dias'], y=df_truchas['pH'],
                  mode='lines', name='pH', line=dict(color='green')),
        row=1, col=2, secondary_y=True
    )
    
    # Lechugas - Crecimiento
    fig.add_trace(
        go.Scatter(x=df_lechugas['Dia'], y=df_lechugas['Altura_cm'],
                  mode='lines+markers', name='Altura Lechugas', line=dict(color='green')),
        row=2, col=1
    )
    
    # Lechugas - Variables ambientales
    fig.add_trace(
        go.Scatter(x=df_lechugas['Dia'], y=df_lechugas['Temperatura_C'],
                  mode='lines', name='Temperatura L', line=dict(color='orange')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df_lechugas['Dia'], y=df_lechugas['Humedad_%'],
                  mode='lines', name='Humedad', line=dict(color='purple')),
        row=2, col=2, secondary_y=True
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Análisis Exploratorio de Datos")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🐟 Modelo Combinado para Truchas")
    
    # Mostrar fórmula
    st.subheader("📐 Fórmula del Modelo Combinado")
    
    st.latex(r"L(t) = L_{\infty} \cdot (1 - e^{-(\beta_0 + \beta_1 \cdot Temp + \beta_2 \cdot O_2 + \beta_3 \cdot Cond + \beta_4 \cdot pH)(t-t_0)})")
    
    st.markdown("""
    **Donde:**
    - L(t): Longitud del pez en el tiempo t
    - L∞: Longitud máxima teórica
    - t₀: Edad teórica en la que el pez tendría longitud cero
    - β₀, β₁, β₂, β₃, β₄: Coeficientes que determinan la tasa de crecimiento k
    - Temp: Temperatura del agua
    - O₂: Oxigenación disuelta
    - Cond: Conductividad eléctrica
    - pH: pH del agua
    """)
    
    try:
        # Datos para el ajuste
        t_data = df_truchas['Tiempo_dias'].values
        L_data = df_truchas['Longitud_cm'].values
        
        # Variables ambientales
        env_vars_truchas = ['Temperatura_C', 'Oxigenacion_mg_L', 'Conductividad_uS_cm', 'pH']
        X_env_truchas = df_truchas[env_vars_truchas].values
        
        # Ajustar modelo combinado
        st.subheader("🔬 Ajuste del Modelo Combinado")
        
        model_result_truchas = fit_combined_model(t_data, L_data, X_env_truchas, max_value_estimate=75)
        L_pred = model_result_truchas['y_pred']
        metrics_truchas = calculate_metrics(L_data, L_pred)
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Parámetros del Modelo")
            st.metric("L∞ (Longitud máxima)", f"{model_result_truchas['L_inf']:.2f} cm")
            st.metric("t₀ (Tiempo inicial)", f"{model_result_truchas['t0']:.2f} días")
            
            st.write("**Coeficientes β (Tasa de crecimiento k):**")
            st.write(f"β₀ (Intercepto): {model_result_truchas['beta_params'][0]:.4f}")
            
            for i, var in enumerate(env_vars_truchas):
                if i + 1 < len(model_result_truchas['beta_params']):
                    coef = model_result_truchas['beta_params'][i + 1]
                    influence = "↗️" if coef > 0 else "↘️"
                    st.write(f"β{i+1} ({var.replace('_', ' ')}): {coef:.4f} {influence}")
        
        with col2:
            st.subheader("📈 Métricas de Evaluación")
            eval_text, emoji = evaluate_r2(metrics_truchas['R²'])
            st.metric("R² (Coeficiente de determinación)", f"{metrics_truchas['R²']:.4f}", delta=f"{eval_text} {emoji}")
            st.metric("MSE (Error Cuadrático Medio)", f"{metrics_truchas['MSE']:.4f}")
            st.metric("MAE (Error Absoluto Medio)", f"{metrics_truchas['MAE']:.4f}")
            st.metric("RMSE (Raíz del Error Cuadrático Medio)", f"{metrics_truchas['RMSE']:.4f}")
            
            # Interpretación del modelo
            st.subheader("🔍 Interpretación del Modelo")
            
            # Calcular k promedio
            k_mean = np.mean(model_result_truchas['k_values'])
            k_min = np.min(model_result_truchas['k_values'])
            k_max = np.max(model_result_truchas['k_values'])
            
            st.write(f"**Tasa de crecimiento k:**")
            st.write(f"- Promedio: {k_mean:.4f}")
            st.write(f"- Mínimo: {k_min:.4f}")
            st.write(f"- Máximo: {k_max:.4f}")
            st.write(f"- Variación: {(k_max - k_min) / k_mean * 100:.1f}%")
        
        # Predicción futura
        st.subheader(f"🔮 Predicción para los próximos {dias_prediccion} días")
        
        # Generar tiempo futuro
        t_future = np.arange(1, max(t_data) + dias_prediccion + 1)
        
        # Usar valores ambientales promedio recientes para predicción
        recent_env = df_truchas.tail(30)[env_vars_truchas].values
        recent_env_mean = recent_env.mean(axis=0).reshape(1, -1)
        
        # Predicción con modelo combinado
        L_future = predict_future(model_result_truchas, t_future, recent_env_mean)
        
        # Gráfico de predicción
        fig_pred = go.Figure()
        
        # Datos históricos
        fig_pred.add_trace(go.Scatter(
            x=t_data, y=L_data,
            mode='markers',
            name='Datos Reales',
            marker=dict(color='blue', size=8)
        ))
        
        # Modelo ajustado
        fig_pred.add_trace(go.Scatter(
            x=t_data, y=L_pred,
            mode='lines',
            name='Modelo Ajustado',
            line=dict(color='red', width=2)
        ))
        
        # Predicción futura
        t_pred_only = t_future[len(t_data):]
        L_pred_only = L_future[len(t_data):]
        
        fig_pred.add_trace(go.Scatter(
            x=t_pred_only, y=L_pred_only,
            mode='lines',
            name='Predicción',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig_pred.update_layout(
            title="Modelo Combinado Von Bertalanffy + Regresión Lineal - Truchas",
            xaxis_title="Tiempo (días)",
            yaxis_title="Longitud (cm)",
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Tabla de predicciones
        pred_df_truchas = pd.DataFrame({
            'Día': t_pred_only,
            'Longitud Predicha (cm)': L_pred_only
        })
        
        st.subheader("📋 Tabla de Predicciones")
        st.dataframe(pred_df_truchas.head(10))
        
        # Análisis de sensibilidad
        st.subheader("🔬 Análisis de Sensibilidad")
        
        # Crear variaciones en las variables ambientales
        variations = {}
        for i, var in enumerate(env_vars_truchas):
            base_value = recent_env_mean[0, i]
            variations[var] = {
                'low': np.copy(recent_env_mean),
                'high': np.copy(recent_env_mean)
            }
            variations[var]['low'][0, i] = base_value * 0.9  # -10%
            variations[var]['high'][0, i] = base_value * 1.1  # +10%
        
        # Calcular predicciones para cada variación
        sensitivity_results = {}
        for var in env_vars_truchas:
            sensitivity_results[var] = {
                'low': predict_future(model_result_truchas, t_future, variations[var]['low']),
                'high': predict_future(model_result_truchas, t_future, variations[var]['high'])
            }
        
        # Gráfico de sensibilidad
        fig_sens = go.Figure()
        
        # Predicción base
        fig_sens.add_trace(go.Scatter(
            x=t_pred_only, y=L_pred_only,
            mode='lines',
            name='Predicción Base',
            line=dict(color='black', width=3)
        ))
        
        # Variaciones
        colors = ['red', 'blue', 'green', 'purple']
        for i, var in enumerate(env_vars_truchas):
            fig_sens.add_trace(go.Scatter(
                x=t_pred_only, 
                y=sensitivity_results[var]['low'][len(t_data):],
                mode='lines',
                name=f"{var.replace('_', ' ')} -10%",
                line=dict(color=colors[i], width=1, dash='dot')
            ))
            
            fig_sens.add_trace(go.Scatter(
                x=t_pred_only, 
                y=sensitivity_results[var]['high'][len(t_data):],
                mode='lines',
                name=f"{var.replace('_', ' ')} +10%",
                line=dict(color=colors[i], width=1)
            ))
        
        fig_sens.update_layout(
            title="Análisis de Sensibilidad - Efecto de Variables Ambientales",
            xaxis_title="Tiempo (días)",
            yaxis_title="Longitud (cm)",
            height=500
        )
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error en el modelo de truchas: {e}")
        import traceback
        st.error(traceback.format_exc())

with tab3:
    st.header("🥬 Modelo Combinado para Lechugas")
    
    # Mostrar fórmulas
    st.subheader("📐 Fórmulas del Modelo Combinado")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Modelo para Altura:**")
        st.latex(r"H(t) = H_{\infty} \cdot (1 - e^{-(\beta_0 + \beta_1 \cdot Temp + \beta_2 \cdot Hum + \beta_3 \cdot pH)(t-t_0)})")
    
    with col2:
        st.markdown("**Modelo para Área Foliar:**")
        st.latex(r"A(t) = A_{\infty} \cdot (1 - e^{-(\beta_0 + \beta_1 \cdot Temp + \beta_2 \cdot Hum + \beta_3 \cdot pH)(t-t_0)})")
    
    st.markdown("""
    **Donde:**
    - H(t): Altura de la lechuga en el tiempo t
    - A(t): Área foliar en el tiempo t
    - H∞, A∞: Valores máximos teóricos
    - t₀: Tiempo teórico inicial
    - β₀, β₁, β₂, β₃: Coeficientes que determinan la tasa de crecimiento k
    - Temp: Temperatura del aire
    - Hum: Humedad relativa
    - pH: pH del sustrato
    """)
    
    try:
        # Datos para el ajuste
        t_data_l = df_lechugas['Dia'].values
        H_data = df_lechugas['Altura_cm'].values
        A_data = df_lechugas['Area_foliar_cm2'].values
        
        # Variables ambientales
        env_vars_lechugas = ['Temperatura_C', 'Humedad_%', 'pH']
        X_env_lechugas = df_lechugas[env_vars_lechugas].values
        
        # 1. MODELO PARA ALTURA
        st.subheader("🔬 1. Modelo Combinado para Altura")
        
        model_result_altura = fit_combined_model(t_data_l, H_data, X_env_lechugas, max_value_estimate=16)
        H_pred = model_result_altura['y_pred']
        metrics_altura = calculate_metrics(H_data, H_pred)
        
        # 2. MODELO PARA ÁREA FOLIAR
        st.subheader("🔬 2. Modelo Combinado para Área Foliar")
        
        model_result_area = fit_combined_model(t_data_l, A_data, X_env_lechugas, max_value_estimate=5000)
        A_pred = model_result_area['y_pred']
        metrics_area = calculate_metrics(A_data, A_pred)
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Resultados - Altura")
            st.metric("H∞ (Altura máxima)", f"{model_result_altura['L_inf']:.2f} cm")
            st.metric("t₀ (Tiempo inicial)", f"{model_result_altura['t0']:.2f} días")
            
            st.write("**Coeficientes β (Tasa de crecimiento k):**")
            st.write(f"β₀ (Intercepto): {model_result_altura['beta_params'][0]:.4f}")
            
            for i, var in enumerate(env_vars_lechugas):
                if i + 1 < len(model_result_altura['beta_params']):
                    coef = model_result_altura['beta_params'][i + 1]
                    influence = "↗️" if coef > 0 else "↘️"
                    st.write(f"β{i+1} ({var.replace('_', ' ')}): {coef:.4f} {influence}")
            
            eval_text_h, emoji_h = evaluate_r2(metrics_altura['R²'])
            st.metric("R²", f"{metrics_altura['R²']:.4f}", delta=f"{eval_text_h} {emoji_h}")
            st.metric("MAE", f"{metrics_altura['MAE']:.4f}")
            st.metric("RMSE", f"{metrics_altura['RMSE']:.4f}")
        
        with col2:
            st.subheader("📊 Resultados - Área Foliar")
            st.metric("A∞ (Área máxima)", f"{model_result_area['L_inf']:.2f} cm²")
            st.metric("t₀ (Tiempo inicial)", f"{model_result_area['t0']:.2f} días")
            
            st.write("**Coeficientes β (Tasa de crecimiento k):**")
            st.write(f"β₀ (Intercepto): {model_result_area['beta_params'][0]:.4f}")
            
            for i, var in enumerate(env_vars_lechugas):
                if i + 1 < len(model_result_area['beta_params']):
                    coef = model_result_area['beta_params'][i + 1]
                    influence = "↗️" if coef > 0 else "↘️"
                    st.write(f"β{i+1} ({var.replace('_', ' ')}): {coef:.4f} {influence}")
            
            eval_text_a, emoji_a = evaluate_r2(metrics_area['R²'])
            st.metric("R²", f"{metrics_area['R²']:.4f}", delta=f"{eval_text_a} {emoji_a}")
            st.metric("MAE", f"{metrics_area['MAE']:.4f}")
            st.metric("RMSE", f"{metrics_area['RMSE']:.4f}")
        
        # Predicción futura
        st.subheader(f"🔮 Predicción para los próximos {dias_prediccion} días")
        
        # Generar tiempo futuro
        t_future_l = np.arange(1, max(t_data_l) + dias_prediccion + 1)
        
        # Usar valores ambientales promedio recientes para predicción
        recent_env_l = df_lechugas.tail(30)[env_vars_lechugas].values
        recent_env_l_mean = recent_env_l.mean(axis=0).reshape(1, -1)
        
        # Predicción con modelo combinado
        H_future = predict_future(model_result_altura, t_future_l, recent_env_l_mean)
        A_future = predict_future(model_result_area, t_future_l, recent_env_l_mean)
        
        # Gráfico de predicción
        fig_pred_l = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Altura', 'Área Foliar')
        )
        
        # Altura
        fig_pred_l.add_trace(
            go.Scatter(x=t_data_l, y=H_data, mode='markers', name='Datos Reales H', 
                      marker=dict(color='blue')), row=1, col=1
        )
        fig_pred_l.add_trace(
            go.Scatter(x=t_data_l, y=H_pred, mode='lines', name='Modelo H', 
                      line=dict(color='red')), row=1, col=1
        )
        
        t_pred_only_l = t_future_l[len(t_data_l):]
        H_pred_only = H_future[len(t_data_l):]
        
        fig_pred_l.add_trace(
            go.Scatter(x=t_pred_only_l, y=H_pred_only, mode='lines', name='Predicción H',
                      line=dict(color='green', dash='dash')), row=1, col=1
        )
        
        # Área foliar
        fig_pred_l.add_trace(
            go.Scatter(x=t_data_l, y=A_data, mode='markers', name='Datos Reales A',
                      marker=dict(color='purple')), row=1, col=2
        )
        fig_pred_l.add_trace(
            go.Scatter(x=t_data_l, y=A_pred, mode='lines', name='Modelo A',
                      line=dict(color='orange')), row=1, col=2
        )
        
        A_pred_only = A_future[len(t_data_l):]
        
        fig_pred_l.add_trace(
            go.Scatter(x=t_pred_only_l, y=A_pred_only, mode='lines', name='Predicción A',
                      line=dict(color='brown', dash='dash')), row=1, col=2
        )
        
        fig_pred_l.update_layout(height=500, title_text="Modelo Combinado - Crecimiento de Lechugas")
        fig_pred_l.update_xaxes(title_text="Tiempo (días)")
        fig_pred_l.update_yaxes(title_text="Altura (cm)", row=1, col=1)
        fig_pred_l.update_yaxes(title_text="Área Foliar (cm²)", row=1, col=2)
        
        st.plotly_chart(fig_pred_l, use_container_width=True)
        
        # Tabla de predicciones
        pred_df_l = pd.DataFrame({
            'Día': t_pred_only_l,
            'Altura Predicha (cm)': H_pred_only,
            'Área Foliar Predicha (cm²)': A_pred_only
        })
        
        st.subheader("📋 Tabla de Predicciones")
        st.dataframe(pred_df_l.head(10))
        
        # Análisis de sensibilidad
        st.subheader("🔬 Análisis de Sensibilidad")
        
        # Crear variaciones en las variables ambientales
        variations_l = {}
        for i, var in enumerate(env_vars_lechugas):
            base_value = recent_env_l_mean[0, i]
            variations_l[var] = {
                'low': np.copy(recent_env_l_mean),
                'high': np.copy(recent_env_l_mean)
            }
            variations_l[var]['low'][0, i] = base_value * 0.9  # -10%
            variations_l[var]['high'][0, i] = base_value * 1.1  # +10%
        
        # Calcular predicciones para cada variación
        sensitivity_results_h = {}
        sensitivity_results_a = {}
        
        for var in env_vars_lechugas:
            sensitivity_results_h[var] = {
                'low': predict_future(model_result_altura, t_future_l, variations_l[var]['low']),
                'high': predict_future(model_result_altura, t_future_l, variations_l[var]['high'])
            }
            
            sensitivity_results_a[var] = {
                'low': predict_future(model_result_area, t_future_l, variations_l[var]['low']),
                'high': predict_future(model_result_area, t_future_l, variations_l[var]['high'])
            }
        
        # Gráficos de sensibilidad
        fig_sens_h = go.Figure()
        fig_sens_a = go.Figure()
        
        # Predicción base - Altura
        fig_sens_h.add_trace(go.Scatter(
            x=t_pred_only_l, y=H_pred_only,
            mode='lines',
            name='Predicción Base',
            line=dict(color='black', width=3)
        ))
        
        # Predicción base - Área
        fig_sens_a.add_trace(go.Scatter(
            x=t_pred_only_l, y=A_pred_only,
            mode='lines',
            name='Predicción Base',
            line=dict(color='black', width=3)
        ))
        
        # Variaciones
        colors = ['red', 'blue', 'green']
        for i, var in enumerate(env_vars_lechugas):
            # Altura
            fig_sens_h.add_trace(go.Scatter(
                x=t_pred_only_l, 
                y=sensitivity_results_h[var]['low'][len(t_data_l):],
                mode='lines',
                name=f"{var.replace('_', ' ')} -10%",
                line=dict(color=colors[i], width=1, dash='dot')
            ))
            
            fig_sens_h.add_trace(go.Scatter(
                x=t_pred_only_l, 
                y=sensitivity_results_h[var]['high'][len(t_data_l):],
                mode='lines',
                name=f"{var.replace('_', ' ')} +10%",
                line=dict(color=colors[i], width=1)
            ))
            
            # Área
            fig_sens_a.add_trace(go.Scatter(
                x=t_pred_only_l, 
                y=sensitivity_results_a[var]['low'][len(t_data_l):],
                mode='lines',
                name=f"{var.replace('_', ' ')} -10%",
                line=dict(color=colors[i], width=1, dash='dot')
            ))
            
            fig_sens_a.add_trace(go.Scatter(
                x=t_pred_only_l, 
                y=sensitivity_results_a[var]['high'][len(t_data_l):],
                mode='lines',
                name=f"{var.replace('_', ' ')} +10%",
                line=dict(color=colors[i], width=1)
            ))
        
        fig_sens_h.update_layout(
            title="Análisis de Sensibilidad - Altura",
            xaxis_title="Tiempo (días)",
            yaxis_title="Altura (cm)",
            height=400
        )
        
        fig_sens_a.update_layout(
            title="Análisis de Sensibilidad - Área Foliar",
            xaxis_title="Tiempo (días)",
            yaxis_title="Área Foliar (cm²)",
            height=400
        )
        
        st.plotly_chart(fig_sens_h, use_container_width=True)
        st.plotly_chart(fig_sens_a, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error en el modelo de lechugas: {e}")
        import traceback
        st.error(traceback.format_exc())

with tab4:
    st.header("📈 Comparación y Métricas Generales")
    
    try:
        # Resumen de métricas
        st.subheader("📊 Resumen de Métricas de Evaluación")
        
        metrics_summary = pd.DataFrame({
            'Modelo': [
                'Truchas - Modelo Combinado',
                'Lechugas Altura - Modelo Combinado',
                'Lechugas Área - Modelo Combinado'
            ],
            'R²': [
                metrics_truchas['R²'],
                metrics_altura['R²'],
                metrics_area['R²']
            ],
            'MAE': [
                metrics_truchas['MAE'],
                metrics_altura['MAE'],
                metrics_area['MAE']
            ],
            'RMSE': [
                metrics_truchas['RMSE'],
                metrics_altura['RMSE'],
                metrics_area['RMSE']
            ]
        })
        
        st.dataframe(metrics_summary.style.format({
            'R²': '{:.4f}',
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}'
        }))
        
        # Gráfico comparativo de R²
        fig_comparison = go.Figure(data=[
            go.Bar(
                x=metrics_summary['Modelo'],
                y=metrics_summary['R²'],
                marker_color=['blue', 'green', 'orange'],
                text=[f"{r2:.3f}" for r2 in metrics_summary['R²']],
                textposition='auto'
            )
        ])
        
        fig_comparison.update_layout(
            title="Comparación de R² entre Modelos",
            xaxis_title="Modelo",
            yaxis_title="R² (Coeficiente de Determinación)",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Interpretación y recomendaciones
        st.subheader("💡 Interpretación y Recomendaciones")
        
        # Evaluar calidad de los modelos
        eval_truchas, _ = evaluate_r2(metrics_truchas['R²'])
        eval_altura, _ = evaluate_r2(metrics_altura['R²'])
        eval_area, _ = evaluate_r2(metrics_area['R²'])
        
        st.write(f"🐟 **Modelo para Truchas:** {eval_truchas} (R² = {metrics_truchas['R²']:.4f})")
        st.write(f"🥬 **Modelo para Altura de Lechugas:** {eval_altura} (R² = {metrics_altura['R²']:.4f})")
        st.write(f"🍃 **Modelo para Área Foliar:** {eval_area} (R² = {metrics_area['R²']:.4f})")
        
        # Recomendaciones basadas en los resultados
        st.subheader("🔍 Recomendaciones")
        
        recommendations = []
        
        # Recomendaciones para truchas
        if metrics_truchas['R²'] < 0.75:
            recommendations.append("🐟 Considerar incluir más variables ambientales para el modelo de truchas")
        else:
            # Identificar variable más influyente
            beta_truchas = model_result_truchas['beta_params'][1:]
            most_influential_idx = np.argmax(np.abs(beta_truchas))
            most_influential_var = env_vars_truchas[most_influential_idx]
            influence = "positiva" if beta_truchas[most_influential_idx] > 0 else "negativa"
            recommendations.append(f"🐟 La variable más influyente en el crecimiento de truchas es {most_influential_var.replace('_', ' ')} (influencia {influence})")
        
        # Recomendaciones para lechugas
        if metrics_altura['R²'] < 0.75 or metrics_area['R²'] < 0.75:
            recommendations.append("🥬 El modelo de lechugas podría beneficiarse de variables adicionales")
        else:
            # Identificar variable más influyente para altura
            beta_altura = model_result_altura['beta_params'][1:]
            most_influential_idx_h = np.argmax(np.abs(beta_altura))
            most_influential_var_h = env_vars_lechugas[most_influential_idx_h]
            influence_h = "positiva" if beta_altura[most_influential_idx_h] > 0 else "negativa"
            
            # Identificar variable más influyente para área
            beta_area = model_result_area['beta_params'][1:]
            most_influential_idx_a = np.argmax(np.abs(beta_area))
            most_influential_var_a = env_vars_lechugas[most_influential_idx_a]
            influence_a = "positiva" if beta_area[most_influential_idx_a] > 0 else "negativa"
            
            recommendations.append(f"🥬 Para altura de lechugas, la variable más influyente es {most_influential_var_h.replace('_', ' ')} (influencia {influence_h})")
            recommendations.append(f"🍃 Para área foliar, la variable más influyente es {most_influential_var_a.replace('_', ' ')} (influencia {influence_a})")
        
        # Mostrar recomendaciones
        for rec in recommendations:
            st.write(rec)
        
        # Fórmulas finales
        st.subheader("📐 Fórmula del Modelo Combinado")
        
        st.markdown("**Fórmula General:**")
        st.latex(r"Y(t) = Y_{\infty} \cdot (1 - e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n)(t-t_0)})")
        
        st.markdown("**Donde:**")
        st.markdown("- Y(t): Variable de crecimiento (longitud, altura, área) en el tiempo t")
        st.markdown("- Y∞: Valor máximo teórico")
        st.markdown("- t₀: Tiempo teórico inicial")
        st.markdown("- β₀, β₁, ..., βₙ: Coeficientes que determinan la tasa de crecimiento k")
        st.markdown("- X₁, X₂, ..., Xₙ: Variables ambientales")
        
        st.markdown("**Métricas de Evaluación:**")
        st.latex(r"R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}")
        st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
        st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")
        
    except Exception as e:
        st.error(f"Error en la comparación: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🐟🥬 Sistema de Predicción Acuapónica</p>
    <p><strong>Modelo Combinado: Von Bertalanffy + Regresión Lineal Multivariante</strong></p>
    <p>L(t) = L∞ · (1 - e^(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)(t-t₀)))</p>
</div>
""", unsafe_allow_html=True)
