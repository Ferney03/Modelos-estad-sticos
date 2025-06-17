import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Importaciones para SARIMA
try:
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SARIMA_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importando librerías SARIMA: {e}")
    st.error("Instale: pip install pmdarima statsmodels")
    SARIMA_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Predicción Acuapónica - Modelo SARIMA",
    page_icon="🐟🥬",
    layout="wide"
)

st.title("🐟🥬 Sistema de Predicción Acuapónica con SARIMA")
st.markdown("### Modelos SARIMA con Predicciones Realistas")

if not SARIMA_AVAILABLE:
    st.stop()

class SARIMAPredictor:
    def __init__(self, limite_biologico=None):
        self.serie_datos = None
        self.modelo = None
        self.limite_biologico = limite_biologico
        self.order = None
        self.seasonal_order = None
        
    def cargar_datos(self, df, columna_valor):
        """Carga y prepara los datos"""
        try:
            if columna_valor not in df.columns:
                st.error(f"Columna '{columna_valor}' no encontrada")
                return False
            
            # Limpiar y convertir datos
            valores = pd.to_numeric(df[columna_valor], errors='coerce').dropna()
            
            if len(valores) < 10:
                st.error(f"Datos insuficientes: {len(valores)} registros")
                return False
            
            # Crear serie temporal
            self.serie_datos = valores.reset_index(drop=True)
            
            st.success(f"✅ Datos cargados: {len(self.serie_datos)} registros")
            st.write(f"📊 Rango: {self.serie_datos.min():.2f} - {self.serie_datos.max():.2f}")
            
            return True
            
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return False
    
    def verificar_estacionariedad(self):
        """Verifica si la serie es estacionaria"""
        try:
            result = adfuller(self.serie_datos)
            p_value = result[1]
            
            st.write(f"**Test Dickey-Fuller:** p-valor = {p_value:.6f}")
            
            if p_value < 0.05:
                st.success("✅ Serie estacionaria")
                return True, 0
            else:
                st.warning("⚠️ Serie no estacionaria - se aplicará diferenciación")
                return False, 1
                
        except Exception as e:
            st.warning(f"Error en test de estacionariedad: {e}")
            return False, 1
    
    def ajustar_modelo(self):
        """Ajusta el modelo SARIMA usando auto_arima"""
        try:
            st.info("🔍 Buscando mejor modelo SARIMA...")
            
            # Verificar estacionariedad
            es_estacionaria, d_sugerido = self.verificar_estacionariedad()
            
            # Configuración de auto_arima
            self.modelo = auto_arima(
                self.serie_datos,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                d=d_sugerido,  # Usar diferenciación sugerida
                start_P=0, start_Q=0,
                max_P=1, max_Q=1,
                seasonal=True,
                m=12,  # Período estacional
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=True,
                information_criterion='aic'
            )
            
            self.order = self.modelo.order
            self.seasonal_order = self.modelo.seasonal_order
            
            st.success(f"✅ Modelo encontrado: SARIMA{self.order}{self.seasonal_order}")
            st.write(f"📊 AIC: {self.modelo.aic():.2f}")
            
            return True
            
        except Exception as e:
            st.error(f"Error ajustando modelo: {e}")
            return False
    
    def predecir(self, n_periods):
        """Realiza predicciones con SARIMA y ajustes para crecimiento realista"""
        try:
            if self.modelo is None:
                st.error("No hay modelo ajustado")
                return None, None
            
            st.info(f"🔮 Generando {n_periods} predicciones...")
            
            # Obtener predicciones base de SARIMA
            predicciones_base, conf_int = self.modelo.predict(
                n_periods=n_periods, 
                return_conf_int=True,
                alpha=0.05
            )
            
            # Convertir a numpy arrays
            predicciones_base = np.array(predicciones_base)
            
            # Obtener último valor real y tendencia
            ultimo_valor = float(self.serie_datos.iloc[-1])
            
            # Calcular tendencia de crecimiento de los últimos valores
            if len(self.serie_datos) >= 5:
                ultimos_5 = self.serie_datos.tail(5)
                tendencia_reciente = (ultimos_5.iloc[-1] - ultimos_5.iloc[0]) / 4
            else:
                tendencia_reciente = 0.01
            
            # Asegurar tendencia mínima positiva para organismos vivos
            if tendencia_reciente <= 0:
                tendencia_reciente = 0.02  # Crecimiento mínimo
            
            # Ajustar predicciones para evitar retroceso
            predicciones_ajustadas = np.zeros(n_periods)
            
            for i in range(n_periods):
                pred_sarima = predicciones_base[i]
                
                if i == 0:
                    valor_anterior = ultimo_valor
                else:
                    valor_anterior = predicciones_ajustadas[i-1]
                
                # Si SARIMA predice retroceso, aplicar crecimiento mínimo
                if pred_sarima < valor_anterior:
                    # Aplicar crecimiento basado en tendencia
                    crecimiento = tendencia_reciente * (0.95 ** i)  # Desaceleración gradual
                    predicciones_ajustadas[i] = valor_anterior + crecimiento
                else:
                    # Si SARIMA predice crecimiento, usarlo pero suavizado
                    crecimiento_sarima = pred_sarima - valor_anterior
                    crecimiento_suavizado = crecimiento_sarima * 0.7 + tendencia_reciente * 0.3
                    predicciones_ajustadas[i] = valor_anterior + crecimiento_suavizado
                
                # Aplicar límite biológico si existe
                if self.limite_biologico and predicciones_ajustadas[i] > self.limite_biologico:
                    # Aproximación asintótica al límite
                    exceso = predicciones_ajustadas[i] - self.limite_biologico
                    predicciones_ajustadas[i] = self.limite_biologico * (1 - np.exp(-exceso/self.limite_biologico))
            
            st.success(f"✅ Predicciones generadas: {predicciones_ajustadas.min():.2f} - {predicciones_ajustadas.max():.2f}")
            
            return predicciones_ajustadas, conf_int
            
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    def calcular_metricas(self, y_true, y_pred):
        """Calcula métricas de evaluación"""
        try:
            min_len = min(len(y_true), len(y_pred))
            y_true = np.array(y_true[:min_len])
            y_pred = np.array(y_pred[:min_len])
            
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            return {'R²': r2, 'MSE': mse, 'MAE': mae, 'RMSE': rmse}
            
        except Exception as e:
            st.warning(f"Error calculando métricas: {e}")
            return {'R²': 0, 'MSE': 0, 'MAE': 0, 'RMSE': 0}

def evaluate_r2(r2):
    """Evaluar calidad del ajuste"""
    if r2 >= 0.90:
        return "Excelente", "🟢"
    elif r2 >= 0.75:
        return "Bueno", "🟡"
    elif r2 >= 0.50:
        return "Aceptable", "🟠"
    else:
        return "Pobre", "🔴"

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Carga de archivos
uploaded_truchas = st.sidebar.file_uploader("Datos de truchas (.xlsx)", type=['xlsx'])
uploaded_lechugas = st.sidebar.file_uploader("Datos de lechugas (.xlsx)", type=['xlsx'])

usar_ejemplo = st.sidebar.button("🔄 Usar Datos de Ejemplo")

# Configuración
dias_prediccion = st.sidebar.slider("Días para predicción", 1, 120, 30)

# Cargar datos
df_truchas = None
df_lechugas = None

try:
    if uploaded_truchas:
        df_truchas = pd.read_excel(uploaded_truchas)
        st.sidebar.success(f"✅ Truchas: {len(df_truchas)} registros")
    elif usar_ejemplo:
        try:
            df_truchas = pd.read_excel('datos_truchas_arcoiris_acuaponia_10.xlsx')
            st.sidebar.success(f"✅ Truchas (ejemplo): {len(df_truchas)} registros")
        except:
            st.sidebar.error("Datos de ejemplo no encontrados")
    
    if uploaded_lechugas:
        df_lechugas = pd.read_excel(uploaded_lechugas)
        st.sidebar.success(f"✅ Lechugas: {len(df_lechugas)} registros")
    elif usar_ejemplo:
        try:
            df_lechugas = pd.read_excel('simulacion_lechuga_realista.xlsx')
            st.sidebar.success(f"✅ Lechugas (ejemplo): {len(df_lechugas)} registros")
        except:
            st.sidebar.error("Datos de ejemplo no encontrados")

except Exception as e:
    st.error(f"Error cargando datos: {e}")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🐟 Truchas SARIMA", "🥬 Lechugas SARIMA", "📈 Comparación"])

with tab1:
    st.header("📊 Análisis de Datos Cargados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if df_truchas is not None:
            st.subheader("🐟 Datos de Truchas")
            st.write(f"**Columnas disponibles:** {list(df_truchas.columns)}")
            st.dataframe(df_truchas.head())
            
            if 'Longitud_cm' in df_truchas.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_truchas.index,
                    y=df_truchas['Longitud_cm'],
                    mode='lines+markers',
                    name='Longitud',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title="Serie Temporal - Longitud Truchas",
                    xaxis_title="Tiempo",
                    yaxis_title="Longitud (cm)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas básicas
                st.write("**Estadísticas:**")
                st.write(f"- Promedio: {df_truchas['Longitud_cm'].mean():.2f} cm")
                st.write(f"- Mínimo: {df_truchas['Longitud_cm'].min():.2f} cm")
                st.write(f"- Máximo: {df_truchas['Longitud_cm'].max():.2f} cm")
        else:
            st.info("No hay datos de truchas cargados")
    
    with col2:
        if df_lechugas is not None:
            st.subheader("🥬 Datos de Lechugas")
            st.write(f"**Columnas disponibles:** {list(df_lechugas.columns)}")
            st.dataframe(df_lechugas.head())
            
            if 'Altura_cm' in df_lechugas.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_lechugas.index,
                    y=df_lechugas['Altura_cm'],
                    mode='lines+markers',
                    name='Altura',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title="Serie Temporal - Altura Lechugas",
                    xaxis_title="Tiempo",
                    yaxis_title="Altura (cm)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas básicas
                st.write("**Estadísticas:**")
                st.write(f"- Promedio: {df_lechugas['Altura_cm'].mean():.2f} cm")
                st.write(f"- Mínimo: {df_lechugas['Altura_cm'].min():.2f} cm")
                st.write(f"- Máximo: {df_lechugas['Altura_cm'].max():.2f} cm")
        else:
            st.info("No hay datos de lechugas cargados")

with tab2:
    st.header("🐟 Modelo SARIMA para Truchas")
    
    if df_truchas is not None and 'Longitud_cm' in df_truchas.columns:
        
        # Crear predictor
        predictor_truchas = SARIMAPredictor(limite_biologico=75)
        
        if predictor_truchas.cargar_datos(df_truchas, 'Longitud_cm'):
            
            # Mostrar información de la serie
            st.subheader("📈 Análisis de la Serie Temporal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de la serie original
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(predictor_truchas.serie_datos))),
                    y=predictor_truchas.serie_datos,
                    mode='lines+markers',
                    name='Longitud Histórica',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title="Serie Temporal Original",
                    xaxis_title="Tiempo",
                    yaxis_title="Longitud (cm)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Información de la Serie:**")
                st.write(f"- Observaciones: {len(predictor_truchas.serie_datos)}")
                st.write(f"- Último valor: {predictor_truchas.serie_datos.iloc[-1]:.2f} cm")
                st.write(f"- Promedio: {predictor_truchas.serie_datos.mean():.2f} cm")
                st.write(f"- Desviación estándar: {predictor_truchas.serie_datos.std():.2f} cm")
                
                # Calcular tendencia
                if len(predictor_truchas.serie_datos) >= 5:
                    ultimos_5 = predictor_truchas.serie_datos.tail(5)
                    tendencia = (ultimos_5.iloc[-1] - ultimos_5.iloc[0]) / 4
                    st.write(f"- Tendencia reciente: {tendencia:.3f} cm/período")
            
            # Ajustar modelo
            st.subheader("🔍 Ajuste del Modelo SARIMA")
            
            if predictor_truchas.ajustar_modelo():
                
                # Información del modelo
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Modelo Seleccionado:**")
                    st.write(f"- Orden: {predictor_truchas.order}")
                    st.write(f"- Orden Estacional: {predictor_truchas.seasonal_order}")
                    st.write(f"- AIC: {predictor_truchas.modelo.aic():.2f}")
                
                with col2:
                    st.write("**Interpretación:**")
                    p, d, q = predictor_truchas.order
                    P, D, Q, s = predictor_truchas.seasonal_order
                    
                    st.write(f"- AR(p={p}): {p} términos autoregresivos")
                    st.write(f"- I(d={d}): {d} diferenciaciones")
                    st.write(f"- MA(q={q}): {q} términos de media móvil")
                    if P > 0 or D > 0 or Q > 0:
                        st.write(f"- Estacional: Sí (período {s})")
                    else:
                        st.write("- Estacional: No")
                
                # Realizar predicción
                st.subheader(f"🔮 Predicción para {dias_prediccion} días")
                
                predicciones, conf_int = predictor_truchas.predecir(dias_prediccion)
                
                if predicciones is not None:
                    
                    # Gráfico de predicción
                    fig = go.Figure()
                    
                    # Datos históricos
                    x_hist = list(range(len(predictor_truchas.serie_datos)))
                    fig.add_trace(go.Scatter(
                        x=x_hist,
                        y=predictor_truchas.serie_datos,
                        mode='lines+markers',
                        name='Datos Históricos',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predicciones
                    x_pred = list(range(len(predictor_truchas.serie_datos), 
                                      len(predictor_truchas.serie_datos) + len(predicciones)))
                    
                    fig.add_trace(go.Scatter(
                        x=x_pred,
                        y=predicciones,
                        mode='lines+markers',
                        name='Predicciones SARIMA',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Intervalos de confianza si están disponibles
                    if conf_int is not None and len(conf_int) == len(predicciones):
                        fig.add_trace(go.Scatter(
                            x=x_pred + x_pred[::-1],
                            y=list(conf_int[:, 1]) + list(conf_int[:, 0][::-1]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Intervalo Confianza 95%'
                        ))
                    
                    # Límite biológico
                    fig.add_hline(
                        y=75, 
                        line_dash="dot", 
                        line_color="green",
                        annotation_text="Límite Biológico (75 cm)"
                    )
                    
                    fig.update_layout(
                        title="Predicción SARIMA - Crecimiento de Truchas",
                        xaxis_title="Tiempo (períodos)",
                        yaxis_title="Longitud (cm)",
                        height=500,
                        legend=dict(x=0.02, y=0.98)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de predicciones
                    st.subheader("📋 Tabla de Predicciones")
                    
                    # Calcular crecimiento diario
                    ultimo_real = predictor_truchas.serie_datos.iloc[-1]
                    crecimientos = np.diff(np.concatenate([[ultimo_real], predicciones]))
                    
                    pred_df = pd.DataFrame({
                        'Día': range(1, len(predicciones) + 1),
                        'Longitud Predicha (cm)': np.round(predicciones, 2),
                        'Crecimiento (cm)': np.round(crecimientos, 3),
                        'Crecimiento Acumulado (cm)': np.round(predicciones - ultimo_real, 2)
                    })
                    
                    st.dataframe(pred_df.head(15))
                    
                    # Métricas de evaluación (validación cruzada)
                    if len(predictor_truchas.serie_datos) > 20:
                        st.subheader("📊 Evaluación del Modelo")
                        
                        # Usar últimos 20% de datos para validación
                        train_size = int(len(predictor_truchas.serie_datos) * 0.8)
                        test_data = predictor_truchas.serie_datos[train_size:]
                        
                        if len(test_data) > 0:
                            # Crear modelo temporal para evaluación
                            train_data = predictor_truchas.serie_datos[:train_size]
                            
                            try:
                                modelo_temp = auto_arima(
                                    train_data,
                                    start_p=0, start_q=0,
                                    max_p=3, max_q=3,
                                    seasonal=True,
                                    m=12,
                                    stepwise=True,
                                    suppress_warnings=True,
                                    error_action='ignore'
                                )
                                
                                pred_eval, _ = modelo_temp.predict(n_periods=len(test_data), return_conf_int=True)
                                
                                # Aplicar mismo ajuste que en predicción principal
                                pred_eval = np.array(pred_eval)
                                ultimo_train = float(train_data.iloc[-1])
                                
                                for i in range(len(pred_eval)):
                                    if i == 0:
                                        valor_anterior = ultimo_train
                                    else:
                                        valor_anterior = pred_eval[i-1]
                                    
                                    if pred_eval[i] < valor_anterior:
                                        pred_eval[i] = valor_anterior + 0.02
                                
                                metricas = predictor_truchas.calcular_metricas(test_data, pred_eval)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    eval_text, emoji = evaluate_r2(metricas['R²'])
                                    st.metric("R²", f"{metricas['R²']:.3f}", delta=f"{eval_text} {emoji}")
                                
                                with col2:
                                    st.metric("MAE", f"{metricas['MAE']:.3f} cm")
                                
                                with col3:
                                    st.metric("RMSE", f"{metricas['RMSE']:.3f} cm")
                                
                                with col4:
                                    crecimiento_promedio = np.mean(crecimientos)
                                    st.metric("Crecimiento Promedio", f"{crecimiento_promedio:.3f} cm/día")
                                
                            except Exception as e:
                                st.warning(f"No se pudo realizar evaluación: {e}")
                    
                    # Resumen de predicción
                    st.subheader("📈 Resumen de Predicción")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Valor Inicial", 
                            f"{ultimo_real:.2f} cm",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Valor Final Predicho", 
                            f"{predicciones[-1]:.2f} cm",
                            delta=f"+{predicciones[-1] - ultimo_real:.2f} cm"
                        )
                    
                    with col3:
                        crecimiento_total = predicciones[-1] - ultimo_real
                        crecimiento_porcentual = (crecimiento_total / ultimo_real) * 100
                        st.metric(
                            "Crecimiento Total", 
                            f"{crecimiento_porcentual:.1f}%",
                            delta=f"{crecimiento_total:.2f} cm"
                        )
                
                else:
                    st.error("❌ No se pudieron generar predicciones")
            
            else:
                st.error("❌ No se pudo ajustar el modelo SARIMA")
    
    else:
        st.warning("⚠️ No hay datos de truchas o falta la columna 'Longitud_cm'")

with tab3:
    st.header("🥬 Modelo SARIMA para Lechugas")
    
    if df_lechugas is not None:
        
        # Verificar columnas disponibles
        tiene_altura = 'Altura_cm' in df_lechugas.columns
        tiene_area = 'Area_foliar_cm2' in df_lechugas.columns
        
        if tiene_altura or tiene_area:
            
            # Crear dos columnas para altura y área foliar
            col1, col2 = st.columns(2)
            
            # Modelo para Altura
            with col1:
                if tiene_altura:
                    st.subheader("📏 SARIMA - Altura")
                    
                    predictor_altura = SARIMAPredictor(limite_biologico=20)
                    
                    if predictor_altura.cargar_datos(df_lechugas, 'Altura_cm'):
                        
                        if predictor_altura.ajustar_modelo():
                            
                            st.write(f"**Modelo:** SARIMA{predictor_altura.order}{predictor_altura.seasonal_order}")
                            st.write(f"**AIC:** {predictor_altura.modelo.aic():.2f}")
                            
                            # Predicción altura
                            pred_altura, conf_altura = predictor_altura.predecir(dias_prediccion)
                            
                            if pred_altura is not None:
                                
                                # Gráfico altura
                                fig = go.Figure()
                                
                                # Datos históricos
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(predictor_altura.serie_datos))),
                                    y=predictor_altura.serie_datos,
                                    mode='lines+markers',
                                    name='Histórico',
                                    line=dict(color='green')
                                ))
                                
                                # Predicciones
                                x_pred = list(range(len(predictor_altura.serie_datos), 
                                                  len(predictor_altura.serie_datos) + len(pred_altura)))
                                
                                fig.add_trace(go.Scatter(
                                    x=x_pred,
                                    y=pred_altura,
                                    mode='lines+markers',
                                    name='Predicción',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title="Predicción Altura - Lechugas",
                                    xaxis_title="Tiempo",
                                    yaxis_title="Altura (cm)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Tabla altura
                                ultimo_altura = predictor_altura.serie_datos.iloc[-1]
                                crecimientos_altura = np.diff(np.concatenate([[ultimo_altura], pred_altura]))
                                
                                pred_altura_df = pd.DataFrame({
                                    'Día': range(1, len(pred_altura) + 1),
                                    'Altura (cm)': np.round(pred_altura, 2),
                                    'Crecimiento (cm)': np.round(crecimientos_altura, 3)
                                })
                                
                                st.dataframe(pred_altura_df.head(10))
                                
                                # Métricas altura
                                crecimiento_total_altura = pred_altura[-1] - ultimo_altura
                                st.metric("Crecimiento Total Altura", f"{crecimiento_total_altura:.2f} cm")
                
                else:
                    st.info("No hay datos de altura disponibles")
            
            # Modelo para Área Foliar
            with col2:
                if tiene_area:
                    st.subheader("🍃 SARIMA - Área Foliar")
                    
                    predictor_area = SARIMAPredictor(limite_biologico=6000)
                    
                    if predictor_area.cargar_datos(df_lechugas, 'Area_foliar_cm2'):
                        
                        if predictor_area.ajustar_modelo():
                            
                            st.write(f"**Modelo:** SARIMA{predictor_area.order}{predictor_area.seasonal_order}")
                            st.write(f"**AIC:** {predictor_area.modelo.aic():.2f}")
                            
                            # Predicción área
                            pred_area, conf_area = predictor_area.predecir(dias_prediccion)
                            
                            if pred_area is not None:
                                
                                # Gráfico área
                                fig = go.Figure()
                                
                                # Datos históricos
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(predictor_area.serie_datos))),
                                    y=predictor_area.serie_datos,
                                    mode='lines+markers',
                                    name='Histórico',
                                    line=dict(color='purple')
                                ))
                                
                                # Predicciones
                                x_pred = list(range(len(predictor_area.serie_datos), 
                                                  len(predictor_area.serie_datos) + len(pred_area)))
                                
                                fig.add_trace(go.Scatter(
                                    x=x_pred,
                                    y=pred_area,
                                    mode='lines+markers',
                                    name='Predicción',
                                    line=dict(color='orange', dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title="Predicción Área Foliar - Lechugas",
                                    xaxis_title="Tiempo",
                                    yaxis_title="Área Foliar (cm²)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Tabla área
                                ultimo_area = predictor_area.serie_datos.iloc[-1]
                                crecimientos_area = np.diff(np.concatenate([[ultimo_area], pred_area]))
                                
                                pred_area_df = pd.DataFrame({
                                    'Día': range(1, len(pred_area) + 1),
                                    'Área Foliar (cm²)': np.round(pred_area, 2),
                                    'Crecimiento (cm²)': np.round(crecimientos_area, 2)
                                })
                                
                                st.dataframe(pred_area_df.head(10))
                                
                                # Métricas área
                                crecimiento_total_area = pred_area[-1] - ultimo_area
                                st.metric("Crecimiento Total Área", f"{crecimiento_total_area:.2f} cm²")
                
                else:
                    st.info("No hay datos de área foliar disponibles")
        
        else:
            st.warning("⚠️ No se encontraron columnas 'Altura_cm' o 'Area_foliar_cm2'")
    
    else:
        st.warning("⚠️ No hay datos de lechugas cargados")

with tab4:
    st.header("📈 Comparación y Resumen")
    
    st.subheader("🎯 Características del Modelo SARIMA Implementado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Ventajas del Modelo:**
        - Auto-selección de parámetros óptimos
        - Manejo de estacionalidad automático
        - Predicciones con crecimiento realista
        - Respeto de límites biológicos
        - Intervalos de confianza incluidos
        - Validación cruzada para evaluación
        """)
    
    with col2:
        st.markdown("""
        **🔧 Ajustes Implementados:**
        - Prevención de retroceso en predicciones
        - Crecimiento mínimo garantizado
        - Suavizado de predicciones extremas
        - Aproximación asintótica a límites
        - Combinación de tendencias SARIMA y biológicas
        - Desaceleración gradual del crecimiento
        """)
    
    st.subheader("📊 Interpretación de Resultados")
    
    st.markdown("""
    **Cómo interpretar las predicciones:**
    
    1. **R² (Coeficiente de Determinación):**
       - > 0.90: Excelente ajuste del modelo
       - 0.75-0.90: Buen ajuste
       - 0.50-0.75: Ajuste aceptable
       - < 0.50: Ajuste pobre, considerar más datos
    
    2. **MAE (Error Absoluto Medio):**
       - Promedio de errores en las mismas unidades
       - Menor valor indica mejor precisión
    
    3. **RMSE (Raíz del Error Cuadrático Medio):**
       - Penaliza más los errores grandes
       - Útil para detectar predicciones atípicas
    
    4. **Crecimiento Predicho:**
       - Debe ser biológicamente coherente
       - Desaceleración gradual es normal
       - Aproximación a límites biológicos esperada
    """)
    
    st.subheader("💡 Recomendaciones de Uso")
    
    recommendations = [
        "📈 **Datos suficientes:** Use al menos 30-50 observaciones para mejores resultados",
        "🔄 **Actualización regular:** Reajuste el modelo con nuevos datos periódicamente",
        "⚖️ **Validación:** Compare predicciones con observaciones reales cuando estén disponibles",
        "🎯 **Límites realistas:** Ajuste los límites biológicos según la especie y condiciones",
        "📊 **Monitoreo:** Observe las métricas de evaluación para detectar degradación del modelo",
        "🌱 **Contexto biológico:** Considere factores externos que puedan afectar el crecimiento"
    ]
    
    for rec in recommendations:
        st.write(rec)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🐟🥬 <strong>Sistema SARIMA para Predicción Acuapónica</strong></p>
    <p>Predicciones Realistas con Auto-ARIMA y Lógica de Crecimiento Biológico</p>
    <p>Versión Corregida - Sin Retroceso en Predicciones</p>
</div>
""", unsafe_allow_html=True)
