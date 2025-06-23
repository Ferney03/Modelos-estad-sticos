import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Suprimir advertencias
warnings.filterwarnings("ignore")

class PredictorPeces:
    def __init__(self):
        self.serie_datos = None
        self.serie_datos_completa = None  # Para almacenar la serie completa para visualización
        self.modelo = None
        self.mejor_modelo_info = None
        self.LIMITE_BIOLOGICO = 70  # Límite biológico en cm para truchas
    
    def cargar_datos(self, archivo):
        """Carga y limpia los datos desde un archivo Excel"""
        try:
            # Intentar cargar el archivo con diferentes engines
            try:
                df = pd.read_excel(archivo, engine='openpyxl')
                print(f"Archivo cargado correctamente con openpyxl: {archivo}")
                print(f"Columnas encontradas: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error con openpyxl: {e}")
                try:
                    df = pd.read_excel(archivo, engine='xlrd')
                    print(f"Archivo cargado correctamente con xlrd: {archivo}")
                    print(f"Columnas encontradas: {df.columns.tolist()}")
                except Exception as e:
                    print(f"Error con xlrd: {e}")
                    return False
            
            # Verificar si existe la columna 'Longitud_cm'
            if 'Longitud_cm' not in df.columns:
                print("No se encontró la columna 'Longitud_cm', buscando alternativas...")
                
                # Buscar columnas que puedan contener datos de longitud
                posibles_columnas = [col for col in df.columns if 'long' in col.lower() or 'cm' in col.lower()]
                if not posibles_columnas:
                    # Si no hay columnas obvias, usar la primera columna numérica
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            print(f"Usando columna numérica: {col}")
                            df['Longitud_cm'] = df[col]
                            break
                else:
                    # Usar la primera columna que parece contener datos de longitud
                    print(f"Usando columna potencial de longitud: {posibles_columnas[0]}")
                    df['Longitud_cm'] = df[posibles_columnas[0]]
            
            # Asegurar que los datos sean numéricos
            df['Longitud_cm'] = pd.to_numeric(df['Longitud_cm'].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Verificar datos antes de limpieza
            print(f"Datos antes de limpieza: {len(df)} registros")
            print(f"Rango de valores: {df['Longitud_cm'].min()} - {df['Longitud_cm'].max()}")
            print(f"Valores NaN: {df['Longitud_cm'].isna().sum()}")
            
            # Eliminar valores NaN
            df = df.dropna(subset=['Longitud_cm'])
            
            # Eliminar outliers y valores incoherentes
            df_original = df.copy()
            df = df[df['Longitud_cm'] <= self.LIMITE_BIOLOGICO]
            df = df[df['Longitud_cm'] > 0]  # Eliminar valores negativos o cero
            
            print(f"Datos después de limpieza: {len(df)} registros")
            print(f"Registros eliminados: {len(df_original) - len(df)}")
            print(f"Rango de valores después de limpieza: {df['Longitud_cm'].min()} - {df['Longitud_cm'].max()}")
            
            # Verificar si hay suficientes datos después de la limpieza
            if len(df) < 10:
                print(f"ADVERTENCIA: Muy pocos datos válidos para el análisis: {len(df)}")
                return False
            
            # Ordenar por índice para asegurar orden temporal
            df = df.sort_index()
            
            # Almacenar la serie completa para visualización
            self.serie_datos_completa = df['Longitud_cm'].copy()
            
            # Almacenar la serie para modelado
            self.serie_datos = df['Longitud_cm']
            
            # Estadísticas básicas para verificación
            print("\nEstadísticas de la serie cargada:")
            print(self.serie_datos.describe())
            
            return True
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualizar_datos(self, serie, titulo, datos_reales=None, predicciones=None):
        """Visualiza datos en un gráfico"""
        try:
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            
            if datos_reales is not None and predicciones is not None:
                # Convertir a arrays de numpy para evitar problemas de índice
                x_reales = np.arange(len(datos_reales))
                x_pred = np.arange(len(datos_reales), len(datos_reales) + len(predicciones))
                
                ax.plot(x_reales, datos_reales.values, label="Datos Reales", color='blue', marker='o', alpha=0.7)
                ax.plot(x_pred, predicciones.values, label="Predicciones", color='red', marker='x', alpha=0.7)
                ax.axhline(y=self.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                           label=f"Límite biológico ({self.LIMITE_BIOLOGICO} cm)")
                ax.legend()
            else:
                ax.plot(np.arange(len(serie)), serie.values, marker='o', linestyle='-', alpha=0.7)
                ax.axhline(y=self.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                           label=f"Límite biológico ({self.LIMITE_BIOLOGICO} cm)")
                ax.legend()
            
            ax.set_title(titulo)
            ax.set_xlabel("Índice")
            ax.set_ylabel("Longitud (cm)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Error al visualizar datos: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def encontrar_mejor_modelo_exponencial(self, serie):
        """Encuentra el mejor modelo de suavizamiento exponencial para la serie temporal"""
        print("\n========== BÚSQUEDA DEL MEJOR MODELO DE SUAVIZAMIENTO EXPONENCIAL ==========")
        
        # Verificar que haya suficientes datos
        if len(serie) < 10:
            raise ValueError(f"La serie de tiempo tiene muy pocos datos para el modelo (tiene {len(serie)}, mínimo requerido 10).")
        
        # Dividir datos para entrenamiento y validación
        train_size = int(len(serie) * 0.8)
        if train_size < 5:
            train_size = int(len(serie) * 0.5)  # Usar al menos la mitad de los datos para train
            
        train = serie[:train_size]
        test = serie[train_size:]
        
        print(f"División de datos: {len(train)} para entrenamiento, {len(test)} para prueba")
        
        modelos = []
        parametros = []
        rmse_valores = []
        aic_valores = []
        mae_valores = []
        r2_valores = []
        
        # 1. Simple Exponential Smoothing (SES) - Para series sin tendencia ni estacionalidad
        try:
            print("\nProbando modelo SES (Simple Exponential Smoothing)...")
            modelo_ses_opt = SimpleExpSmoothing(train).fit(optimized=True)
            forecast_ses_opt = modelo_ses_opt.forecast(len(test))
            
            # Verificar las predicciones
            print(f"Predicciones SES: {forecast_ses_opt.values}")
            
            # Evaluar
            rmse = np.sqrt(mean_squared_error(test, forecast_ses_opt))
            mae = mean_absolute_error(test, forecast_ses_opt)
            r2 = r2_score(test, forecast_ses_opt)
            aic = modelo_ses_opt.aic
            
            modelos.append(modelo_ses_opt)
            parametros.append(f"SES_optimizado(alpha={modelo_ses_opt.params['smoothing_level']:.4f})")
            rmse_valores.append(rmse)
            mae_valores.append(mae)
            r2_valores.append(r2)
            aic_valores.append(aic)
            
            print(f"SES optimizado: alpha={modelo_ses_opt.params['smoothing_level']:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, AIC={aic:.4f}")
        except Exception as e:
            print(f"Error en SES optimizado: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Modelo Holt (para tendencia)
        try:
            print("\nProbando modelo Holt (tendencia)...")
            modelo_holt_opt = Holt(train).fit(optimized=True)
            forecast_holt_opt = modelo_holt_opt.forecast(len(test))
            
            # Verificar las predicciones
            print(f"Predicciones Holt: {forecast_holt_opt.values}")
            
            # Evaluar
            rmse = np.sqrt(mean_squared_error(test, forecast_holt_opt))
            mae = mean_absolute_error(test, forecast_holt_opt)
            r2 = r2_score(test, forecast_holt_opt)
            aic = modelo_holt_opt.aic
            
            modelos.append(modelo_holt_opt)
            alpha_opt = modelo_holt_opt.params.get('smoothing_level', 'N/A')
            beta_opt = modelo_holt_opt.params.get('smoothing_slope', 'N/A')
            parametros.append(f"Holt_optimizado(alpha={alpha_opt}, beta={beta_opt})")
            rmse_valores.append(rmse)
            mae_valores.append(mae)
            r2_valores.append(r2)
            aic_valores.append(aic)
            
            print(f"Holt optimizado: alpha={alpha_opt}, beta={beta_opt}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, AIC={aic:.4f}")
        except Exception as e:
            print(f"Error en Holt optimizado: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Modelo Holt con tendencia amortiguada
        try:
            print("\nProbando modelo Holt con tendencia amortiguada...")
            modelo_holt_damped = Holt(train, damped_trend=True).fit(optimized=True)
            forecast_holt_damped = modelo_holt_damped.forecast(len(test))
            
            # Verificar las predicciones
            print(f"Predicciones Holt amortiguado: {forecast_holt_damped.values}")
            
            # Evaluar
            rmse = np.sqrt(mean_squared_error(test, forecast_holt_damped))
            mae = mean_absolute_error(test, forecast_holt_damped)
            r2 = r2_score(test, forecast_holt_damped)
            aic = modelo_holt_damped.aic
            
            modelos.append(modelo_holt_damped)
            alpha_opt = modelo_holt_damped.params.get('smoothing_level', 'N/A')
            beta_opt = modelo_holt_damped.params.get('smoothing_slope', 'N/A')
            phi = modelo_holt_damped.params.get('damping_slope', 'N/A')
            parametros.append(f"Holt_amortiguado(alpha={alpha_opt}, beta={beta_opt}, phi={phi})")
            rmse_valores.append(rmse)
            mae_valores.append(mae)
            r2_valores.append(r2)
            aic_valores.append(aic)
            
            print(f"Holt amortiguado: alpha={alpha_opt}, beta={beta_opt}, phi={phi}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, AIC={aic:.4f}")
        except Exception as e:
            print(f"Error en Holt amortiguado: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Si hay suficientes datos, probar Holt-Winters (para estacionalidad)
        seasonal_periods = [4, 7, 12]  # Diferentes períodos estacionales
        
        if len(train) >= 24:  # Al menos 2 temporadas
            for seasonal_period in seasonal_periods:
                if len(train) >= 2 * seasonal_period:  # Asegurar suficientes ciclos
                    # Modelo aditivo
                    try:
                        print(f"\nProbando modelo Holt-Winters aditivo (período={seasonal_period})...")
                        modelo_hw_add = ExponentialSmoothing(
                            train, 
                            seasonal_periods=seasonal_period,
                            trend='add',
                            seasonal='add',
                            damped_trend=True
                        ).fit(optimized=True, remove_bias=True)
                        
                        forecast_hw_add = modelo_hw_add.forecast(len(test))
                        
                        # Verificar las predicciones
                        print(f"Predicciones HW aditivo: {forecast_hw_add.values}")
                        
                        # Evaluar
                        rmse = np.sqrt(mean_squared_error(test, forecast_hw_add))
                        mae = mean_absolute_error(test, forecast_hw_add)
                        r2 = r2_score(test, forecast_hw_add)
                        aic = modelo_hw_add.aic
                        
                        modelos.append(modelo_hw_add)
                        parametros.append(f"HoltWinters_aditivo(sp={seasonal_period})")
                        rmse_valores.append(rmse)
                        mae_valores.append(mae)
                        r2_valores.append(r2)
                        aic_valores.append(aic)
                        
                        print(f"Holt-Winters aditivo (sp={seasonal_period}): RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, AIC={aic:.4f}")
                    except Exception as e:
                        print(f"Error en Holt-Winters aditivo (sp={seasonal_period}): {e}")
                    
                    # Modelo multiplicativo
                    try:
                        print(f"\nProbando modelo Holt-Winters multiplicativo (período={seasonal_period})...")
                        modelo_hw_mul = ExponentialSmoothing(
                            train, 
                            seasonal_periods=seasonal_period,
                            trend='add',
                            seasonal='mul',
                            damped_trend=True
                        ).fit(optimized=True, remove_bias=True)
                        
                        forecast_hw_mul = modelo_hw_mul.forecast(len(test))
                        
                        # Verificar las predicciones
                        print(f"Predicciones HW multiplicativo: {forecast_hw_mul.values}")
                        
                        # Evaluar
                        rmse = np.sqrt(mean_squared_error(test, forecast_hw_mul))
                        mae = mean_absolute_error(test, forecast_hw_mul)
                        r2 = r2_score(test, forecast_hw_mul)
                        aic = modelo_hw_mul.aic
                        
                        modelos.append(modelo_hw_mul)
                        parametros.append(f"HoltWinters_multiplicativo(sp={seasonal_period})")
                        rmse_valores.append(rmse)
                        mae_valores.append(mae)
                        r2_valores.append(r2)
                        aic_valores.append(aic)
                        
                        print(f"Holt-Winters multiplicativo (sp={seasonal_period}): RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, AIC={aic:.4f}")
                    except Exception as e:
                        print(f"Error en Holt-Winters multiplicativo (sp={seasonal_period}): {e}")
        
        # Seleccionar el mejor modelo basado en el menor RMSE
        if rmse_valores:
            idx_mejor = np.argmin(rmse_valores)
            self.modelo = modelos[idx_mejor]
            mejor_parametro = parametros[idx_mejor]
            mejor_rmse = rmse_valores[idx_mejor]
            mejor_mae = mae_valores[idx_mejor]
            mejor_r2 = r2_valores[idx_mejor]
            mejor_aic = aic_valores[idx_mejor]
            
            print(f"\nMejor modelo: {mejor_parametro}")
            print(f"RMSE: {mejor_rmse:.4f}, MAE: {mejor_mae:.4f}, R²: {mejor_r2:.4f}, AIC: {mejor_aic:.4f}")
            
            # Guardar información del modelo
            self.mejor_modelo_info = {
                "tipo": mejor_parametro,
                "rmse": mejor_rmse,
                "mae": mejor_mae,
                "r2": mejor_r2,
                "aic": mejor_aic,
                "parametros": self.modelo.params
            }
            
            return self.modelo
        else:
            # Si ningún modelo funciona, usar un SES básico con parámetros conservadores
            print("No se pudo ajustar ningún modelo optimizado. Usando modelo básico.")
            modelo_basico = SimpleExpSmoothing(train).fit(smoothing_level=0.3, optimized=False)
            self.modelo = modelo_basico
            self.mejor_modelo_info = {
                "tipo": "SES básico (alpha=0.3)",
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "aic": modelo_basico.aic if hasattr(modelo_basico, 'aic') else np.nan,
                "parametros": modelo_basico.params
            }
            return modelo_basico
    
    def evaluar_prediccion(self, datos_reales, predicciones):
        """Evalúa la calidad de las predicciones con múltiples métricas"""
        try:
            # Verificar si hay datos para evaluar
            if len(datos_reales) == 0 or len(predicciones) == 0:
                print("No hay suficientes datos para evaluar la predicción")
                return np.nan, np.nan, np.nan, np.nan
            
            # Asegurar que ambos arrays tienen la misma longitud
            min_len = min(len(datos_reales), len(predicciones))
            datos_reales = datos_reales[:min_len]
            predicciones = predicciones[:min_len]
            
            # Imprimir algunos valores para diagnóstico
            print("\nEvaluación de predicciones:")
            print(f"Datos reales (primeros 5): {datos_reales[:5].values}")
            print(f"Predicciones (primeros 5): {predicciones[:5].values}")
            
            # Error Cuadrático Medio (MSE)
            mse = mean_squared_error(datos_reales, predicciones)
            
            # Raíz del Error Cuadrático Medio (RMSE)
            rmse = np.sqrt(mse)
            
            # Error Absoluto Medio (MAE)
            mae = mean_absolute_error(datos_reales, predicciones)
            
            # R Cuadrado (R²)
            r2 = r2_score(datos_reales, predicciones)
            
            print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            return mse, rmse, mae, r2
        except Exception as e:
            print(f"Error al evaluar predicción: {e}")
            import traceback
            traceback.print_exc()
            return np.nan, np.nan, np.nan, np.nan
    
    def predecir_longitud(self, dias_a_predecir):
        """Realiza la predicción de longitud para los días especificados"""
        if self.serie_datos is None:
            raise ValueError("Debe cargar datos primero.")
        
        # Asegurar que la serie esté limpia y sea numérica
        serie = self.serie_datos.copy()
        serie = serie.dropna().astype(float)
        
        if len(serie) < 10:
            raise ValueError(f"La serie de tiempo tiene muy pocos datos para el modelo (tiene {len(serie)}, mínimo requerido 10).")
        
        # Encontrar el mejor modelo de suavizamiento exponencial
        self.encontrar_mejor_modelo_exponencial(serie)
        
        if self.modelo is None:
            raise ValueError("No se pudo ajustar un modelo de suavizamiento exponencial.")
        
        # Mostrar información del modelo seleccionado
        print(f"\n============= RESUMEN DEL MODELO SELECCIONADO =============")
        print(f"Modelo seleccionado: {self.mejor_modelo_info['tipo']}")
        print(f"Parámetros: {self.mejor_modelo_info['parametros']}")
        
        if not np.isnan(self.mejor_modelo_info['rmse']):
            print(f"RMSE en validación: {self.mejor_modelo_info['rmse']:.4f}")
        
        if not np.isnan(self.mejor_modelo_info['mae']):
            print(f"MAE en validación: {self.mejor_modelo_info['mae']:.4f}")
        
        if not np.isnan(self.mejor_modelo_info['r2']):
            print(f"R² en validación: {self.mejor_modelo_info['r2']:.4f}")
        
        if not np.isnan(self.mejor_modelo_info['aic']):
            print(f"AIC: {self.mejor_modelo_info['aic']:.4f}")
        
        print("===========================================================\n")
        
        try:
            # Reentrenar el modelo con todos los datos disponibles
            if "SES" in self.mejor_modelo_info['tipo']:
                # Para modelos SES
                if "optimizado" in self.mejor_modelo_info['tipo']:
                    modelo_final = SimpleExpSmoothing(serie).fit(optimized=True)
                else:
                    alpha = float(self.mejor_modelo_info['parametros'].get('smoothing_level', 0.3))
                    modelo_final = SimpleExpSmoothing(serie).fit(smoothing_level=alpha, optimized=False)
            
            elif "Holt" in self.mejor_modelo_info['tipo'] and "Winters" not in self.mejor_modelo_info['tipo']:
                # Para modelos Holt
                if "amortiguado" in self.mejor_modelo_info['tipo']:
                    modelo_final = Holt(serie, damped_trend=True).fit(optimized=True)
                else:
                    modelo_final = Holt(serie).fit(optimized=True)
            
            elif "HoltWinters" in self.mejor_modelo_info['tipo']:
                # Para modelos Holt-Winters
                # Extraer período estacional del nombre del modelo
                import re
                match = re.search(r'sp=(\d+)', self.mejor_modelo_info['tipo'])
                if match:
                    sp = int(match.group(1))
                else:
                    sp = 12  # Valor por defecto
                
                seasonal_type = 'add'
                if "multiplicativo" in self.mejor_modelo_info['tipo']:
                    seasonal_type = 'mul'
                
                modelo_final = ExponentialSmoothing(
                    serie,
                    seasonal_periods=sp,
                    trend='add',
                    seasonal=seasonal_type,
                    damped_trend=True
                ).fit(optimized=True, remove_bias=True)
            else:
                # Modelo por defecto
                modelo_final = SimpleExpSmoothing(serie).fit(smoothing_level=0.3, optimized=False)
            
            # Generar predicciones
            predicciones_originales = modelo_final.forecast(dias_a_predecir)
            print(f"\nPredicciones originales del modelo (primeros 5 días): {predicciones_originales[:5].values}")
            
        except Exception as e:
            print(f"Error al reentrenar modelo: {e}")
            import traceback
            traceback.print_exc()
            
            # Plan B: usar el modelo original
            try:
                predicciones_originales = self.modelo.forecast(dias_a_predecir)
            except Exception as e:
                print(f"Error al usar modelo original: {e}")
                # Plan C: crear un modelo básico
                modelo_emergencia = SimpleExpSmoothing(serie).fit(smoothing_level=0.3, optimized=False)
                predicciones_originales = modelo_emergencia.forecast(dias_a_predecir)
        
        # Crear un índice para las predicciones
        indices_pred = pd.RangeIndex(start=len(serie), stop=len(serie) + dias_a_predecir)
        predicciones = pd.Series(predicciones_originales, index=indices_pred)
        
        # Guardar copia de las predicciones originales antes de aplicar restricciones
        predicciones_sin_restricciones = predicciones.copy()
        
        # Aplicar restricciones biológicas con menor impacto
        ultimo_valor_real = serie.iloc[-1]
        
        for i in range(len(predicciones)):
            # Inicializar con el valor anterior (ya sea el último real o la predicción anterior)
            valor_anterior = ultimo_valor_real if i == 0 else predicciones.iloc[i-1]
            
            # 1. Restricción de decrecimiento moderada: permite pequeñas disminuciones
            if predicciones.iloc[i] < valor_anterior * 0.95:  # Permite hasta 5% de disminución
                predicciones.iloc[i] = valor_anterior * 0.95  # Limita la disminución a 5%
            
            # 2. Limitar al límite biológico
            if predicciones.iloc[i] > self.LIMITE_BIOLOGICO:
                predicciones.iloc[i] = self.LIMITE_BIOLOGICO
            
            # 3. Desacelerar crecimiento cerca del límite biológico (solo si está muy cerca)
            if predicciones.iloc[i] > self.LIMITE_BIOLOGICO * 0.98:  # Solo si está al 98% del límite
                distancia_al_limite = self.LIMITE_BIOLOGICO - valor_anterior
                crecimiento = distancia_al_limite * 0.1  # 10% de la distancia restante al límite
                predicciones.iloc[i] = min(self.LIMITE_BIOLOGICO, valor_anterior + crecimiento)
        
        # Verificar el impacto de las restricciones biológicas
        if len(predicciones) > 0:
            cambios = (predicciones - predicciones_sin_restricciones).abs()
            print(f"\nImpacto de restricciones biológicas:")
            print(f"Cambio medio: {cambios.mean():.4f} cm")
            print(f"Cambio máximo: {cambios.max():.4f} cm")
            print(f"Número de valores modificados: {(cambios > 0.001).sum()} de {len(predicciones)}")
        
        # Verificar las predicciones finales
        print(f"\nPredicciones finales (primeros 5 días): {predicciones[:5].values}")
        
        return predicciones

# Interfaz gráfica mejorada
class InterfazPredictor:
    def __init__(self, root):
        self.root = root
        self.predictor = PredictorPeces()
        self.configurar_interfaz()
    
    def configurar_interfaz(self):
        """Configura los elementos de la interfaz gráfica"""
        self.root.title("Predicción de Crecimiento de Peces - Suavizamiento Exponencial")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f4f8")
        
        # Crear marco principal
        main_frame = tk.Frame(self.root, bg="#f0f4f8", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        titulo = tk.Label(main_frame, text="Sistema de Predicción de Crecimiento de Truchas", 
                         font=("Arial", 18, "bold"), bg="#f0f4f8", fg="#1a5276")
        titulo.pack(pady=10)
        
        # Subtítulo
        subtitulo = tk.Label(main_frame, 
                           text=f"Suavizamiento Exponencial Adaptativo (Límite biológico: {self.predictor.LIMITE_BIOLOGICO} cm)", 
                           font=("Arial", 12), bg="#f0f4f8", fg="#2874a6")
        subtitulo.pack(pady=5)
        
        # Marco para entrada de datos
        datos_frame = tk.LabelFrame(main_frame, text="Carga de datos", bg="#e1e8ed", 
                                  font=("Arial", 12, "bold"), fg="#34495e", padx=15, pady=15)
        datos_frame.pack(fill=tk.X, pady=10)
        
        # Botón para cargar archivo
        self.btn_cargar = tk.Button(datos_frame, text="Cargar Archivo Excel", command=self.cargar_archivo, 
                               font=("Arial", 12), bg="#3498db", fg="white", padx=15, pady=5)
        self.btn_cargar.pack(pady=10)
        
        # Etiqueta de archivo seleccionado
        self.lbl_archivo = tk.Label(datos_frame, text="Ningún archivo seleccionado", 
                               font=("Arial", 10), bg="#e1e8ed", fg="#566573")
        self.lbl_archivo.pack(pady=5)
        
        # Entrada para días
        dias_frame = tk.Frame(datos_frame, bg="#e1e8ed")
        dias_frame.pack(pady=10)
        
        tk.Label(dias_frame, text="Días a predecir:", font=("Arial", 12), bg="#e1e8ed").pack(side=tk.LEFT, padx=5)
        self.entry_dias = tk.Entry(dias_frame, font=("Arial", 12), width=10, bd=2)
        self.entry_dias.pack(side=tk.LEFT, padx=5)
        self.entry_dias.insert(0, "30")
        
        # Botón de predicción
        self.btn_predecir = tk.Button(main_frame, text="Realizar Predicción", command=self.realizar_prediccion, 
                                    font=("Arial", 14, "bold"), bg="#2ecc71", fg="white", padx=20, pady=10)
        self.btn_predecir.pack(pady=15)
        
        # Crear notebook (pestañas)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Pestaña de estadísticas
        self.tab_estadisticas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_estadisticas, text="Estadísticas del Modelo")
        
        # Pestaña de resultados
        self.tab_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_resultados, text="Resultados de Predicción")
        
        # Pestaña de gráficos
        self.tab_graficos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_graficos, text="Gráficos")
        
        # Pestaña de tabla de crecimiento
        self.tab_tabla = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tabla, text="Tabla de Crecimiento")
        
        # Configurar contenido de las pestañas
        self.configurar_tab_estadisticas()
        self.configurar_tab_resultados()
        self.configurar_tab_graficos()
        self.configurar_tab_tabla()
        
        # Barra de estado
        self.status_bar = tk.Label(self.root, text="Listo", bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                              font=("Arial", 10), bg="#d6eaf8", fg="#2c3e50")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def configurar_tab_estadisticas(self):
        """Configura la pestaña de estadísticas"""
        frame = tk.Frame(self.tab_estadisticas, bg="#f5f5f5", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        tk.Label(frame, text="Estadísticas del Modelo", font=("Arial", 14, "bold"), 
               bg="#f5f5f5", fg="#34495e").pack(pady=5)
        
        # Área para mostrar estadísticas
        self.estadisticas_text = tk.Text(frame, font=("Consolas", 11), height=20, width=80)
        self.estadisticas_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar para el área de texto
        scrollbar = tk.Scrollbar(self.estadisticas_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.estadisticas_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.estadisticas_text.yview)
    
    def configurar_tab_resultados(self):
        """Configura la pestaña de resultados"""
        frame = tk.Frame(self.tab_resultados, bg="#f5f5f5", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        tk.Label(frame, text="Resultados de Predicción", font=("Arial", 14, "bold"), 
               bg="#f5f5f5", fg="#34495e").pack(pady=5)
        
        # Área para mostrar resultados
        self.resultado_text = tk.Text(frame, font=("Consolas", 11), height=20, width=80)
        self.resultado_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar para el área de texto
        scrollbar = tk.Scrollbar(self.resultado_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.resultado_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.resultado_text.yview)
    
    def configurar_tab_graficos(self):
        """Configura la pestaña de gráficos"""
        frame = tk.Frame(self.tab_graficos, bg="#f5f5f5", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        tk.Label(frame, text="Visualización de Datos y Predicciones", font=("Arial", 14, "bold"), 
               bg="#f5f5f5", fg="#34495e").pack(pady=5)
        
        # Frame para los gráficos
        self.graficos_frame = tk.Frame(frame, bg="#f5f5f5")
        self.graficos_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial
        tk.Label(self.graficos_frame, text="Realice una predicción para ver los gráficos", 
               font=("Arial", 12), bg="#f5f5f5", fg="#7f8c8d").pack(pady=100)
    
    def configurar_tab_tabla(self):
        """Configura la pestaña de tabla de crecimiento"""
        frame = tk.Frame(self.tab_tabla, bg="#f5f5f5", padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        tk.Label(frame, text="Tabla de Crecimiento Diario", font=("Arial", 14, "bold"), 
               bg="#f5f5f5", fg="#34495e").pack(pady=5)
        
        # Crear tabla con Treeview
        columns = ("Día", "Longitud (cm)", "Crecimiento (cm)", "Estado")
        self.tabla_tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        
        # Configurar encabezados
        for col in columns:
            self.tabla_tree.heading(col, text=col)
            self.tabla_tree.column(col, width=150, anchor="center")
        
        # Scrollbar para la tabla
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tabla_tree.yview)
        self.tabla_tree.configure(yscrollcommand=scrollbar.set)
        
        # Colocar tabla y scrollbar
        self.tabla_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def cargar_archivo(self):
        """Maneja la carga de archivo"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de datos", 
            filetypes=[("Archivos Excel", "*.xlsx *.xls"), ("Todos los archivos", "*.*")]
        )
        
        if not archivo:
            return
        
        self.archivo_path = archivo
        self.lbl_archivo.config(text=f"Archivo: {archivo.split('/')[-1]}")
        self.status_bar.config(text="Archivo seleccionado correctamente")
    
    def realizar_prediccion(self):
        """Realiza el proceso de predicción"""
        if not hasattr(self, 'archivo_path'):
            messagebox.showerror("Error", "Debe seleccionar un archivo primero.")
            return

        try:
            dias = self.entry_dias.get()
            if not dias.isdigit() or int(dias) <= 0:
                messagebox.showerror("Error", "Ingrese un número válido de días mayor a 0.")
                return
        
            dias = int(dias)
            self.status_bar.config(text="Cargando datos...")
            self.root.update()
        
            # Limpiar áreas de texto y gráficos
            self.estadisticas_text.delete(1.0, tk.END)
            self.resultado_text.delete(1.0, tk.END)
            
            # Limpiar tabla
            for item in self.tabla_tree.get_children():
                self.tabla_tree.delete(item)
            
            # Limpiar gráficos
            for widget in self.graficos_frame.winfo_children():
                widget.destroy()
            
            # Cargar datos
            if not self.predictor.cargar_datos(self.archivo_path):
                messagebox.showerror("Error", "No se pudieron cargar los datos correctamente. Verifique el formato del archivo.")
                self.status_bar.config(text="Error al cargar datos")
                return
        
            # Redireccionar salida a variable
            import io
            import sys
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Realizar predicción
            try:
                predicciones = self.predictor.predecir_longitud(dias)
                
                # Restaurar stdout
                sys.stdout = old_stdout
                output = new_stdout.getvalue()
                
                # Mostrar información estadística
                self.mostrar_estadisticas(output)
                
                # Evaluar con datos de prueba si hay suficientes
                serie = self.predictor.serie_datos
                train_size = int(len(serie) * 0.8)
                test = serie[train_size:]
                
                if len(test) > 0:
                    n_comparar = min(dias, len(test))
                    test_compare = test[:n_comparar]
                    predicciones_comparar = predicciones[:n_comparar]
                    
                    if len(test_compare) > 0:
                        mse, rmse, mae, r2 = self.predictor.evaluar_prediccion(
                            test_compare, predicciones_comparar)
                        self.mostrar_resultados(predicciones, mse, rmse, mae, r2)
                    else:
                        self.mostrar_resultados(predicciones)
                else:
                    self.mostrar_resultados(predicciones)
                
                # Mostrar gráficos
                self.mostrar_graficos(serie, predicciones)
                
                # Mostrar tabla de crecimiento
                self.mostrar_tabla_crecimiento(predicciones)
                
                self.status_bar.config(text="Predicción completada exitosamente")
                self.notebook.select(self.tab_resultados)  # Cambiar a la pestaña de resultados
                
            except Exception as e:
                sys.stdout = old_stdout
                import traceback
                error_msg = f"Error durante la predicción: {str(e)}"
                error_details = traceback.format_exc()
                print(f"Error detallado: {error_details}")
                
                messagebox.showerror("Error", error_msg)
                self.status_bar.config(text="Error en la predicción")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {str(e)}")
            self.status_bar.config(text="Error inesperado")
    
    def mostrar_estadisticas(self, output):
        """Muestra la información estadística en el área de texto correspondiente"""
        self.estadisticas_text.delete(1.0, tk.END)
        
        # Información de encabezado
        self.estadisticas_text.insert(tk.END, "ANÁLISIS ESTADÍSTICO DEL MODELO\n")
        self.estadisticas_text.insert(tk.END, "==========================================\n\n")
        
        # Extraer información relevante de la salida capturada
        lines = output.split('\n')
        
        # Buscar secciones específicas
        found_model_info = False
        
        for i, line in enumerate(lines):
            if "RESUMEN DEL MODELO SELECCIONADO" in line:
                found_model_info = True
                # Extraer las siguientes líneas con información del modelo
                self.estadisticas_text.insert(tk.END, "INFORMACIÓN DEL MODELO:\n")
                self.estadisticas_text.insert(tk.END, "==========================================\n")
                continue
            
            if found_model_info and "===========" in line:
                found_model_info = False  # Stop capturing when separator line is found
            
            if found_model_info:
                self.estadisticas_text.insert(tk.END, line + "\n")
        
        # Añadir interpretación
        self.estadisticas_text.insert(tk.END, "\nINTERPRETACIÓN DE MÉTRICAS:\n")
        self.estadisticas_text.insert(tk.END, "==========================================\n")
        self.estadisticas_text.insert(tk.END, "- RMSE (Raíz del Error Cuadrático Medio): Mide la desviación estándar de los residuos.\n")
        self.estadisticas_text.insert(tk.END, "  Valores más bajos indican mejor ajuste del modelo.\n\n")
        self.estadisticas_text.insert(tk.END, "- MAE (Error Absoluto Medio): Mide el promedio de los errores absolutos.\n")
        self.estadisticas_text.insert(tk.END, "  Menos sensible a valores atípicos que RMSE.\n\n")
        self.estadisticas_text.insert(tk.END, "- R² (Coeficiente de Determinación): Indica la proporción de varianza explicada por el modelo.\n")
        self.estadisticas_text.insert(tk.END, "  Valores cercanos a 1 indican mejor ajuste.\n\n")
        self.estadisticas_text.insert(tk.END, "- AIC (Criterio de Información de Akaike): Mide la calidad relativa del modelo.\n")
        self.estadisticas_text.insert(tk.END, "  Valores más bajos indican mejor equilibrio entre ajuste y complejidad.\n\n")
        
        self.estadisticas_text.insert(tk.END, "TIPOS DE MODELOS DE SUAVIZAMIENTO EXPONENCIAL:\n")
        self.estadisticas_text.insert(tk.END, "==========================================\n")
        self.estadisticas_text.insert(tk.END, "- SES (Simple): Para series sin tendencia ni estacionalidad.\n")
        self.estadisticas_text.insert(tk.END, "- Holt: Para series con tendencia.\n")
        self.estadisticas_text.insert(tk.END, "- Holt amortiguado: Para series con tendencia que se desacelera.\n")
        self.estadisticas_text.insert(tk.END, "- Holt-Winters: Para series con tendencia y estacionalidad.\n")
    
    def mostrar_resultados(self, predicciones, mse=None, rmse=None, mae=None, r2=None):
        """Muestra los resultados de la predicción en el área de texto correspondiente"""
        self.resultado_text.delete(1.0, tk.END)
        
        # Encabezado de resultados
        self.resultado_text.insert(tk.END, "RESULTADOS DE PREDICCIÓN\n")
        self.resultado_text.insert(tk.END, "==========================================\n\n")
        
        # Mostrar métricas de evaluación si están disponibles
        if mse is not None and rmse is not None and mae is not None and r2 is not None:
            self.resultado_text.insert(tk.END, "Métricas de Evaluación:\n")
            self.resultado_text.insert(tk.END, "-----------------------------------------\n")
            self.resultado_text.insert(tk.END, f"MSE (Error Cuadrático Medio): {mse:.4f}\n")
            self.resultado_text.insert(tk.END, f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}\n")
            self.resultado_text.insert(tk.END, f"MAE (Error Absoluto Medio): {mae:.4f}\n")
            self.resultado_text.insert(tk.END, f"R² (Coeficiente de Determinación): {r2:.4f}\n\n")
        
        # Mostrar resumen de predicciones
        self.resultado_text.insert(tk.END, "Resumen de Predicciones:\n")
        self.resultado_text.insert(tk.END, "-----------------------------------------\n")
        self.resultado_text.insert(tk.END, f"Días predichos: {len(predicciones)}\n")
        self.resultado_text.insert(tk.END, f"Valor inicial: {predicciones.iloc[0]:.2f} cm\n")
        self.resultado_text.insert(tk.END, f"Valor final: {predicciones.iloc[-1]:.2f} cm\n")
        
        # Calcular crecimiento total
        crecimiento_total = predicciones.iloc[-1] - predicciones.iloc[0]
        self.resultado_text.insert(tk.END, f"Crecimiento total: {crecimiento_total:.2f} cm\n")
        self.resultado_text.insert(tk.END, f"Crecimiento promedio diario: {crecimiento_total/len(predicciones):.4f} cm/día\n\n")
        
        # Mostrar predicciones detalladas
        self.resultado_text.insert(tk.END, "Predicciones Detalladas:\n")
        self.resultado_text.insert(tk.END, "-----------------------------------------\n")
        self.resultado_text.insert(tk.END, "Día\tLongitud (cm)\tCrecimiento (cm)\n")
        
        for i, valor in enumerate(predicciones):
            crecimiento = 0 if i == 0 else valor - predicciones.iloc[i-1]
            self.resultado_text.insert(tk.END, f"{i+1}\t{valor:.2f}\t\t{'+' if i > 0 else ''}{crecimiento:.4f}\n")
    
    def mostrar_graficos(self, serie, predicciones):
        """Muestra los gráficos de datos y predicciones"""
        # Limpiar frame de gráficos
        for widget in self.graficos_frame.winfo_children():
            widget.destroy()
        
        # Crear frame para los gráficos
        graficos_container = tk.Frame(self.graficos_frame, bg="#f5f5f5")
        graficos_container.pack(fill=tk.BOTH, expand=True)
        
        # 1. Gráfico de serie temporal y predicciones
        fig1 = Figure(figsize=(8, 4))
        ax1 = fig1.add_subplot(111)
        
        # Graficar serie original
        ax1.plot(range(len(serie)), serie.values, 'b-', marker='o', markersize=4, label='Datos históricos')
        
        # Graficar predicciones
        ax1.plot(range(len(serie), len(serie) + len(predicciones)), predicciones.values, 'r-', marker='x', markersize=4, label='Predicciones')
        
        # Añadir línea de límite biológico
        ax1.axhline(y=self.predictor.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                   label=f"Límite biológico ({self.predictor.LIMITE_BIOLOGICO} cm)")
        
        ax1.set_title('Serie Temporal y Predicciones')
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Longitud (cm)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Añadir el gráfico al frame
        canvas1 = FigureCanvasTkAgg(fig1, master=graficos_container)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # 2. Gráfico de crecimiento diario
        fig2 = Figure(figsize=(8, 4))
        ax2 = fig2.add_subplot(111)
        
        # Calcular crecimientos diarios
        crecimientos = []
        for i in range(1, len(predicciones)):
            crecimiento = predicciones.iloc[i] - predicciones.iloc[i-1]
            crecimientos.append(crecimiento)
        
        # Graficar crecimiento diario
        ax2.bar(range(1, len(crecimientos) + 1), crecimientos, color='green', alpha=0.7)
        ax2.set_title('Tasa de Crecimiento Diario Predicho')
        ax2.set_xlabel('Día')
        ax2.set_ylabel('Crecimiento (cm/día)')
        ax2.grid(True, alpha=0.3)
        
        # Añadir el gráfico al frame
        canvas2 = FigureCanvasTkAgg(fig2, master=graficos_container)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
    
    def mostrar_tabla_crecimiento(self, predicciones):
        """Muestra la tabla de crecimiento diario"""
        # Limpiar tabla
        for item in self.tabla_tree.get_children():
            self.tabla_tree.delete(item)
        
        # Llenar tabla con datos de predicción
        for i, valor in enumerate(predicciones):
            # Calcular crecimiento desde el día anterior
            crecimiento = 0
            if i > 0:
                crecimiento = valor - predicciones.iloc[i-1]
            
            # Determinar el estado del crecimiento
            estado = ""
            
            if i == 0:
                estado = "Valor inicial"
            elif valor >= self.predictor.LIMITE_BIOLOGICO * 0.95:
                estado = "¡Cerca del límite biológico!"
            elif crecimiento > 0.1:
                estado = "Crecimiento rápido"
            elif crecimiento < 0.01:
                estado = "Crecimiento lento"
            else:
                estado = "Crecimiento normal"
            
            # Insertar fila en la tabla
            self.tabla_tree.insert("", "end", values=(
                f"{i+1}",
                f"{valor:.2f}",
                f"{'+' if i > 0 else ''}{crecimiento:.4f}" if i > 0 else "-",
                estado
            ))
            
            # Aplicar color según el estado
            item_id = self.tabla_tree.get_children()[-1]
            if estado == "¡Cerca del límite biológico!":
                self.tabla_tree.item(item_id, tags=("limite",))
            elif estado == "Crecimiento rápido":
                self.tabla_tree.item(item_id, tags=("rapido",))
            elif estado == "Crecimiento lento":
                self.tabla_tree.item(item_id, tags=("lento",))
            elif estado == "Valor inicial":
                self.tabla_tree.item(item_id, tags=("inicial",))
        
        # Configurar colores para los tags
        self.tabla_tree.tag_configure("limite", background="#f8d7da")
        self.tabla_tree.tag_configure("rapido", background="#d4edda")
        self.tabla_tree.tag_configure("lento", background="#fff3cd")
        self.tabla_tree.tag_configure("inicial", background="#e8f4f8")

# Punto de entrada principal
if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazPredictor(root)
    root.mainloop()