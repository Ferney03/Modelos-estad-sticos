import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize as optimize

class ModeloCrecimiento:
    """
    Modelo matemático para predecir el crecimiento de organismos (peces o lechugas)
    utilizando el modelo de Von Bertalanffy con ajustes para factores ambientales.
    """
    
    def __init__(self, tipo_organismo='pez'):
        """
        Inicializa el modelo de crecimiento
        
        Args:
            tipo_organismo (str): 'pez' o 'lechuga'
        """
        self.tipo_organismo = tipo_organismo.lower()
        self.data = None
        self.modelo_lineal = None
        self.modelo_von_bertalanffy = None
        self.parametros_vb = None
        self.max_day = 0
        self.feature_names = []
        self.p_values = []
        self.parametros_limite = self.obtener_parametros_limite()
        
    def obtener_parametros_limite(self):
        """Define los límites biológicos según el tipo de organismo"""
        if self.tipo_organismo == 'pez':
            return {
                'crecimiento_diario_max': 0.5,  # cm por día
                'crecimiento_diario_min': 0,    # cm por día (no puede decrecer)
                'tamaño_maximo': 70,            # cm (para truchas)
                'unidad': 'cm',
                'L_inf_default': 60,            # Longitud asintótica por defecto
                'K_default': 0.01,              # Coeficiente de crecimiento por defecto
                't0_default': -0.5              # Edad teórica a longitud cero por defecto
            }
        elif self.tipo_organismo == 'lechuga':
            return {
                'crecimiento_diario_max': 1.2,  # cm por día
                'crecimiento_diario_min': 0,    # cm por día (no puede decrecer)
                'tamaño_maximo': 30,            # cm (para lechugas)
                'unidad': 'cm',
                'L_inf_default': 25,            # Longitud asintótica por defecto
                'K_default': 0.08,              # Coeficiente de crecimiento por defecto
                't0_default': -0.2              # Edad teórica a longitud cero por defecto
            }
        else:
            raise ValueError("Tipo de organismo no soportado. Use 'pez' o 'lechuga'")
    
    def cargar_datos(self, datos):
        """
        Carga y prepara los datos para el entrenamiento
        
        Args:
            datos (DataFrame o lista): Datos de crecimiento con columnas 'Tiempo_dias' y 'Longitud_cm'
            
        Returns:
            DataFrame: Datos procesados
        """
        try:
            # Convertir a DataFrame si es una lista
            if isinstance(datos, list):
                self.data = pd.DataFrame(datos)
            else:
                self.data = datos.copy()
            
            # Verificar columnas requeridas
            columnas_requeridas = ["Tiempo_dias", "Longitud_cm"]
            for col in columnas_requeridas:
                if col not in self.data.columns:
                    raise ValueError(f"No se encontró la columna requerida '{col}' en los datos")
            
            # Reemplazar comas con puntos en columnas numéricas y convertir a float
            for col in self.data.columns:
                if self.data[col].dtype == object:
                    self.data[col] = self.data[col].astype(str).str.replace(',', '.').astype(float)
            
            # Ordenar datos por días para asegurar orden cronológico
            self.data = self.data.sort_values(by="Tiempo_dias")
            
            # Obtener el día máximo de los datos
            self.max_day = self.data["Tiempo_dias"].max()
            
            # Obtener nombres de características (todas las columnas excepto Longitud_cm)
            self.feature_names = [col for col in self.data.columns if col != "Longitud_cm"]
            
            # Calcular tasas históricas de crecimiento
            self.data["Growth_Rate"] = self.data["Longitud_cm"].diff() / self.data["Tiempo_dias"].diff()
            
            print(f"Datos cargados: {len(self.data)} registros")
            return self.data
            
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            raise
    
    def modelo_von_bertalanffy_func(self, t, L_inf, K, t0):
        """
        Función del modelo de crecimiento de Von Bertalanffy
        
        Args:
            t (float): Tiempo en días
            L_inf (float): Longitud asintótica (longitud máxima teórica)
            K (float): Coeficiente de crecimiento (tasa a la que se alcanza L_inf)
            t0 (float): Edad teórica cuando la longitud sería cero
            
        Returns:
            float: Longitud predicha en el tiempo t
        """
        return L_inf * (1 - np.exp(-K * (t - t0)))
    
    def ajustar_modelo_von_bertalanffy(self):
        """
        Ajusta el modelo de Von Bertalanffy a los datos
        
        Returns:
            tuple: Parámetros optimizados (L_inf, K, t0)
        """
        try:
            # Datos para el ajuste
            t = self.data["Tiempo_dias"].values
            longitud = self.data["Longitud_cm"].values
            
            # Valores iniciales para los parámetros
            p0 = [
                self.parametros_limite['L_inf_default'],
                self.parametros_limite['K_default'],
                self.parametros_limite['t0_default']
            ]
            
            # Límites para los parámetros
            bounds = (
                [longitud.max(), 0, -10],  # límites inferiores
                [self.parametros_limite['tamaño_maximo'] * 1.5, 1, 0]  # límites superiores
            )
            
            # Función de error a minimizar
            def error_func(p, t, longitud):
                return self.modelo_von_bertalanffy_func(t, *p) - longitud
            
            # Optimización
            resultado = optimize.least_squares(
                error_func, p0, args=(t, longitud), bounds=bounds
            )
            
            # Extraer parámetros optimizados
            L_inf, K, t0 = resultado.x
            
            # Guardar parámetros
            self.parametros_vb = {
                'L_inf': L_inf,
                'K': K,
                't0': t0
            }
            
            # Calcular R²
            longitud_pred = self.modelo_von_bertalanffy_func(t, L_inf, K, t0)
            ss_total = np.sum((longitud - np.mean(longitud))**2)
            ss_residual = np.sum((longitud - longitud_pred)**2)
            r2 = 1 - (ss_residual / ss_total)
            
            # Calcular RMSE
            rmse = np.sqrt(np.mean((longitud - longitud_pred)**2))
            
            # Guardar métricas
            self.metricas_vb = {
                'r2': r2,
                'rmse': rmse
            }
            
            print(f"Modelo Von Bertalanffy ajustado: L_inf={L_inf:.2f}, K={K:.4f}, t0={t0:.2f}")
            print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}")
            
            return self.parametros_vb
            
        except Exception as e:
            print(f"Error al ajustar modelo Von Bertalanffy: {str(e)}")
            # Usar valores por defecto si falla el ajuste
            self.parametros_vb = {
                'L_inf': self.parametros_limite['L_inf_default'],
                'K': self.parametros_limite['K_default'],
                't0': self.parametros_limite['t0_default']
            }
            return self.parametros_vb
    
    def entrenar_modelo_lineal(self):
        """
        Entrena un modelo de regresión lineal para factores ambientales
        
        Returns:
            LinearRegression: Modelo entrenado
        """
        try:
            print("Entrenando modelo lineal para factores ambientales...")
            
            # Preparar características (X) y objetivo (y)
            # Usamos solo las características ambientales (no el tiempo)
            features_ambientales = [f for f in self.feature_names if f != "Tiempo_dias"]
            
            if not features_ambientales:
                print("No hay características ambientales disponibles")
                return None
                
            X = self.data[features_ambientales]
            
            # El objetivo es la desviación del modelo VB
            # Primero calculamos las predicciones del modelo VB
            if not self.parametros_vb:
                self.ajustar_modelo_von_bertalanffy()
                
            vb_pred = self.modelo_von_bertalanffy_func(
                self.data["Tiempo_dias"].values,
                self.parametros_vb['L_inf'],
                self.parametros_vb['K'],
                self.parametros_vb['t0']
            )
            
            # La desviación es la diferencia entre el valor real y la predicción VB
            y = self.data["Longitud_cm"].values - vb_pred
            
            # Crear y entrenar el modelo de scikit-learn
            self.modelo_lineal = LinearRegression()
            self.modelo_lineal.fit(X, y)
            
            # Calcular métricas en datos de entrenamiento
            y_pred = self.modelo_lineal.predict(X)
            self.mse = mean_squared_error(y, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y, y_pred)
            self.r2 = r2_score(y, y_pred)
            
            # Usar statsmodels para obtener valores p
            X_sm = sm.add_constant(X)
            model_sm = sm.OLS(y, X_sm).fit()
            
            # Extraer valores p
            self.p_values = model_sm.pvalues
            
            print(f"Modelo lineal entrenado: R² = {self.r2:.4f}, RMSE = {self.rmse:.4f}")
            
            # Imprimir coeficientes y su significancia
            print("\nCoeficientes del modelo lineal:")
            print(f"Intercepto: {self.modelo_lineal.intercept_:.6f}")
            
            for i, feature in enumerate(features_ambientales):
                coef = self.modelo_lineal.coef_[i]
                p_value = self.p_values.get(feature, 1.0)
                significance = ""
                if p_value < 0.001:
                    significance = "***"  # Altamente significativo
                elif p_value < 0.01:
                    significance = "**"   # Muy significativo
                elif p_value < 0.05:
                    significance = "*"    # Significativo
                elif p_value < 0.1:
                    significance = "."    # Marginalmente significativo
                
                print(f"{feature}: {coef:.6f} (p-valor: {p_value:.6f}) {significance}")
            
            return self.modelo_lineal
            
        except Exception as e:
            print(f"Error al entrenar el modelo lineal: {str(e)}")
            return None
    
    def get_historical_growth_rate(self, analysis_period=30):
        """
        Calcula estadísticas de la tasa de crecimiento histórica
        
        Args:
            analysis_period (int): Número de días para analizar
            
        Returns:
            dict: Estadísticas de crecimiento
        """
        try:
            # Obtener los puntos de datos más recientes basados en el periodo de análisis
            recent_data = self.data.tail(analysis_period)
            
            # Calcular la tasa de crecimiento diario promedio
            total_growth = recent_data["Longitud_cm"].iloc[-1] - recent_data["Longitud_cm"].iloc[0]
            total_days = recent_data["Tiempo_dias"].iloc[-1] - recent_data["Tiempo_dias"].iloc[0]
            
            if total_days == 0:
                return {
                    "avg_rate": 0,
                    "std_rate": 0,
                    "max_rate": 0,
                    "min_rate": 0,
                    "recent_rate": 0
                }
                
            avg_daily_growth = total_growth / total_days
            
            # Calcular la desviación estándar del crecimiento diario
            daily_growth_rates = recent_data["Growth_Rate"].dropna()
            std_daily_growth = daily_growth_rates.std() if len(daily_growth_rates) > 1 else 0
            
            return {
                "avg_rate": avg_daily_growth,
                "std_rate": std_daily_growth,
                "max_rate": daily_growth_rates.max() if len(daily_growth_rates) > 0 else avg_daily_growth,
                "min_rate": daily_growth_rates.min() if len(daily_growth_rates) > 0 else 0,
                "recent_rate": daily_growth_rates.iloc[-5:].mean() if len(daily_growth_rates) >= 5 else avg_daily_growth
            }
        except Exception as e:
            print(f"Error al calcular la tasa histórica de crecimiento: {str(e)}")
            return {
                "avg_rate": 0.01, 
                "std_rate": 0.005, 
                "max_rate": 0.02, 
                "min_rate": 0, 
                "recent_rate": 0.01
            }
    
    def predecir_crecimiento(self, dias_predecir, parametros_futuros=None, tipo_restriccion='historica'):
        """
        Predice el crecimiento futuro
        
        Args:
            dias_predecir (int): Número de días a predecir
            parametros_futuros (dict): Parámetros ambientales futuros
            tipo_restriccion (str): Tipo de restricción ('historica' o 'suavizada')
            
        Returns:
            DataFrame: Predicciones de crecimiento
        """
        try:
            print(f"Prediciendo valores futuros para {dias_predecir} días...")
            
            # Verificar que los modelos estén entrenados
            if not self.parametros_vb:
                self.ajustar_modelo_von_bertalanffy()
            
            # Obtener la última fila de datos como punto de partida
            last_row = self.data.iloc[-1:].copy()
            last_length = last_row["Longitud_cm"].values[0]
            
            # Obtener estadísticas de tasa histórica de crecimiento
            growth_stats = self.get_historical_growth_rate()
            
            # Crear un dataframe para almacenar predicciones
            future_data = pd.DataFrame(columns=self.data.columns)
            
            # Añadir el último punto de datos conocido
            future_data = pd.concat([future_data, last_row])
            
            # Si no se proporcionan parámetros futuros, usar los últimos valores
            if parametros_futuros is None:
                parametros_futuros = {}
                for feature in self.feature_names:
                    if feature != "Tiempo_dias":
                        parametros_futuros[feature] = last_row[feature].values[0]
            
            # Generar predicciones para días futuros
            for i in range(1, dias_predecir + 1):
                new_day = self.max_day + i
                new_row = last_row.copy()
                new_row["Tiempo_dias"] = new_day
                
                # Actualizar parámetros ambientales
                for feature, value in parametros_futuros.items():
                    if feature in new_row.columns:
                        new_row[feature] = value
                
                # Predecir longitud usando el modelo de Von Bertalanffy
                vb_prediction = self.modelo_von_bertalanffy_func(
                    new_day,
                    self.parametros_vb['L_inf'],
                    self.parametros_vb['K'],
                    self.parametros_vb['t0']
                )
                
                # Ajustar predicción con factores ambientales si el modelo lineal está disponible
                env_adjustment = 0
                if self.modelo_lineal is not None:
                    # Extraer características ambientales
                    features_ambientales = [f for f in self.feature_names if f != "Tiempo_dias"]
                    if features_ambientales:
                        X_env = new_row[features_ambientales]
                        env_adjustment = self.modelo_lineal.predict(X_env)[0]
                
                # Predicción ajustada del modelo
                model_prediction = vb_prediction + env_adjustment
                
                if tipo_restriccion == "historica":
                    # Usar tasa histórica de crecimiento con pequeña variación aleatoria
                    random_factor = np.random.normal(0, growth_stats["std_rate"] / 3)
                    growth_rate = growth_stats["recent_rate"] + random_factor
                    
                    # Asegurar que la tasa de crecimiento esté dentro de límites históricos
                    growth_rate = max(growth_stats["min_rate"], min(growth_stats["max_rate"], growth_rate))
                    
                    # Calcular nueva longitud basada en tasa histórica de crecimiento
                    new_length = last_length + growth_rate
                    
                    # Mezclar con predicción del modelo (90% histórico, 10% modelo)
                    predicted_length = 0.9 * new_length + 0.1 * model_prediction
                    
                elif tipo_restriccion == "suavizada":
                    # Usar suavizado exponencial entre tasa histórica y predicción del modelo
                    model_growth = model_prediction - last_length
                    historical_growth = growth_stats["recent_rate"]
                    
                    # Si el crecimiento del modelo es demasiado alto comparado con el histórico, reducir su influencia
                    if abs(model_growth) > 3 * abs(historical_growth):
                        alpha = 0.1  # Bajo peso para el modelo
                    else:
                        alpha = 0.3  # Peso moderado para el modelo
                    
                    # Mezclar tasas de crecimiento
                    blended_growth = alpha * model_growth + (1 - alpha) * historical_growth
                    predicted_length = last_length + blended_growth
                
                else:  # modelo puro
                    predicted_length = model_prediction
                
                # Aplicar restricción realista según el tipo de organismo
                predicted_length = min(predicted_length, self.parametros_limite['tamaño_maximo'])
                
                # Asegurar que el organismo no encoja
                predicted_length = max(predicted_length, last_length)
                
                new_row["Longitud_cm"] = predicted_length
                new_row["Predicted_Length"] = predicted_length
                
                # Calcular tasa de crecimiento
                new_row["Growth_Rate"] = (predicted_length - last_length)
                
                # Añadir la nueva fila a future_data
                future_data = pd.concat([future_data, new_row])
                
                # Actualizar last_row y last_length para la siguiente iteración
                last_row = new_row.copy()
                last_length = predicted_length
            
            print("Predicción completada")
            return future_data.iloc[1:]  # Omitir la primera fila (que era el último punto de datos conocido)
        
        except Exception as e:
            print(f"Error al predecir valores futuros: {str(e)}")
            return None
    
    def visualizar_resultados(self, future_data, mostrar_grafico=True):
        """
        Visualiza los resultados de la predicción
        
        Args:
            future_data (DataFrame): Datos de predicción
            mostrar_grafico (bool): Si se debe mostrar el gráfico
            
        Returns:
            dict: Resumen de resultados
        """
        try:
            # Preparar resumen de resultados
            resumen = {
                "tipo_organismo": self.tipo_organismo,
                "parametros_vb": self.parametros_vb,
                "metricas_vb": getattr(self, 'metricas_vb', {}),
                "metricas_lineal": {
                    "r2": getattr(self, 'r2', 0),
                    "rmse": getattr(self, 'rmse', 0),
                    "mse": getattr(self, 'mse', 0),
                    "mae": getattr(self, 'mae', 0)
                },
                "predicciones": []
            }
            
            # Añadir predicciones al resumen
            for _, row in future_data.iterrows():
                resumen["predicciones"].append({
                    "dia": int(row["Tiempo_dias"]),
                    "longitud": float(row["Longitud_cm"]),
                    "tasa_crecimiento": float(row["Growth_Rate"])
                })
            
            # Mostrar resumen en consola
            print("\nResumen de predicción:")
            print(f"Tipo de organismo: {self.tipo_organismo}")
            print(f"Parámetros Von Bertalanffy: L_inf={self.parametros_vb['L_inf']:.2f}, K={self.parametros_vb['K']:.4f}, t0={self.parametros_vb['t0']:.2f}")
            
            if hasattr(self, 'metricas_vb'):
                print(f"Métricas Von Bertalanffy: R²={self.metricas_vb['r2']:.4f}, RMSE={self.metricas_vb['rmse']:.4f}")
            
            if hasattr(self, 'r2'):
                print(f"Métricas modelo lineal: R²={self.r2:.4f}, RMSE={self.rmse:.4f}")
            
            print("\nPredicciones de crecimiento:")
            print(f"{'Día':<8}{'Longitud (cm)':<15}{'Crecimiento diario (cm)':<25}")
            print("-" * 48)
            
            # Mostrar cada 5 días y el último
            for i, pred in enumerate(resumen["predicciones"]):
                if i % 5 == 0 or i == len(resumen["predicciones"]) - 1:
                    print(f"{pred['dia']:<8}{pred['longitud']:<15.4f}{pred['tasa_crecimiento']:<25.6f}")
            
            # Mostrar predicción final
            ultimo_dia = resumen["predicciones"][-1]["dia"]
            ultima_longitud = resumen["predicciones"][-1]["longitud"]
            print(f"\nEn el día {ultimo_dia}, la longitud predicha será {ultima_longitud:.4f} cm")
            
            # Crear gráfico si se solicita
            if mostrar_grafico:
                plt.figure(figsize=(10, 6))
                
                # Graficar datos originales
                plt.scatter(self.data["Tiempo_dias"], self.data["Longitud_cm"], 
                           color='blue', label='Datos Originales', s=30, alpha=0.7)
                
                # Graficar datos futuros predichos
                plt.scatter(future_data["Tiempo_dias"], future_data["Longitud_cm"], 
                           color='red', label='Datos Predichos', s=30, alpha=0.7)
                
                # Conectar los puntos con líneas
                all_days = list(self.data["Tiempo_dias"]) + list(future_data["Tiempo_dias"])
                all_lengths = list(self.data["Longitud_cm"]) + list(future_data["Longitud_cm"])
                plt.plot(all_days, all_lengths, 'k--', alpha=0.5)
                
                # Graficar curva de Von Bertalanffy
                t_range = np.linspace(0, max(all_days), 100)
                vb_curve = self.modelo_von_bertalanffy_func(
                    t_range,
                    self.parametros_vb['L_inf'],
                    self.parametros_vb['K'],
                    self.parametros_vb['t0']
                )
                plt.plot(t_range, vb_curve, 'g-', label='Modelo Von Bertalanffy', alpha=0.7)
                
                plt.xlabel('Tiempo (días)')
                plt.ylabel(f'Longitud ({self.parametros_limite["unidad"]})')
                plt.title(f'Predicción de Crecimiento de {self.tipo_organismo.capitalize()}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()
            
            return resumen
            
        except Exception as e:
            print(f"Error al visualizar resultados: {str(e)}")
            return None

# Función para predecir crecimiento (para uso en API)
def predecir_crecimiento(tipo_organismo, datos, dias_predecir, parametros_futuros=None, tipo_restriccion='historica'):
    """
    Función para predecir el crecimiento de un organismo
    
    Args:
        tipo_organismo (str): 'pez' o 'lechuga'
        datos (list o DataFrame): Datos históricos de crecimiento
        dias_predecir (int): Número de días a predecir
        parametros_futuros (dict): Parámetros ambientales futuros
        tipo_restriccion (str): Tipo de restricción ('historica', 'suavizada' o 'modelo')
        
    Returns:
        dict: Resultados de la predicción
    """
    try:
        # Crear modelo
        modelo = ModeloCrecimiento(tipo_organismo)
        
        # Cargar datos
        modelo.cargar_datos(datos)
        
        # Ajustar modelo de Von Bertalanffy
        modelo.ajustar_modelo_von_bertalanffy()
        
        # Entrenar modelo lineal para factores ambientales
        modelo.entrenar_modelo_lineal()
        
        # Predecir crecimiento
        predicciones = modelo.predecir_crecimiento(dias_predecir, parametros_futuros, tipo_restriccion)
        
        # Visualizar resultados (sin mostrar gráfico)
        resultados = modelo.visualizar_resultados(predicciones, mostrar_grafico=False)
        
        return resultados
        
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return {
            "error": str(e),
            "tipo_organismo": tipo_organismo,
            "predicciones": []
        }

# Ejemplo de uso para peces
def ejemplo_peces():
    print("\n=== EJEMPLO DE PREDICCIÓN PARA PECES ===")
    
    # Datos de ejemplo (tiempo en días, longitud en cm)
    datos_peces = [
        {"Tiempo_dias": 0, "Longitud_cm": 5.0, "pH": 7.2, "Temperatura": 18},
        {"Tiempo_dias": 7, "Longitud_cm": 5.8, "pH": 7.1, "Temperatura": 18.5},
        {"Tiempo_dias": 14, "Longitud_cm": 6.7, "pH": 7.3, "Temperatura": 19},
        {"Tiempo_dias": 21, "Longitud_cm": 7.9, "pH": 7.0, "Temperatura": 19.2},
        {"Tiempo_dias": 28, "Longitud_cm": 9.2, "pH": 7.2, "Temperatura": 18.8},
        {"Tiempo_dias": 35, "Longitud_cm": 10.6, "pH": 7.4, "Temperatura": 19.5},
        {"Tiempo_dias": 42, "Longitud_cm": 12.1, "pH": 7.3, "Temperatura": 19.8},
        {"Tiempo_dias": 49, "Longitud_cm": 13.8, "pH": 7.1, "Temperatura": 20},
        {"Tiempo_dias": 56, "Longitud_cm": 15.5, "pH": 7.2, "Temperatura": 19.7},
        {"Tiempo_dias": 63, "Longitud_cm": 17.3, "pH": 7.3, "Temperatura": 19.5},
    ]
    
    # Crear modelo
    modelo_peces = ModeloCrecimiento('pez')
    
    # Cargar datos
    modelo_peces.cargar_datos(datos_peces)
    
    # Ajustar modelo de Von Bertalanffy
    modelo_peces.ajustar_modelo_von_bertalanffy()
    
    # Entrenar modelo lineal para factores ambientales
    modelo_peces.entrenar_modelo_lineal()
    
    # Predecir crecimiento para los próximos 30 días
    parametros_futuros = {
        "pH": 7.2,
        "Temperatura": 19.5
    }
    
    predicciones_peces = modelo_peces.predecir_crecimiento(30, parametros_futuros, 'historica')
    
    # Visualizar resultados
    modelo_peces.visualizar_resultados(predicciones_peces)

# Ejemplo de uso para lechugas
def ejemplo_lechugas():
    print("\n=== EJEMPLO DE PREDICCIÓN PARA LECHUGAS ===")
    
    # Datos de ejemplo (tiempo en días, longitud en cm)
    datos_lechugas = [
        {"Tiempo_dias": 0, "Longitud_cm": 2.0, "pH": 6.5, "Temperatura": 22},
        {"Tiempo_dias": 3, "Longitud_cm": 3.2, "pH": 6.6, "Temperatura": 22.5},
        {"Tiempo_dias": 6, "Longitud_cm": 4.7, "pH": 6.4, "Temperatura": 23},
        {"Tiempo_dias": 9, "Longitud_cm": 6.5, "pH": 6.5, "Temperatura": 22.8},
        {"Tiempo_dias": 12, "Longitud_cm": 8.6, "pH": 6.7, "Temperatura": 22.5},
        {"Tiempo_dias": 15, "Longitud_cm": 11.0, "pH": 6.6, "Temperatura": 23.2},
        {"Tiempo_dias": 18, "Longitud_cm": 13.8, "pH": 6.5, "Temperatura": 23.5},
        {"Tiempo_dias": 21, "Longitud_cm": 16.9, "pH": 6.4, "Temperatura": 23.0},
        {"Tiempo_dias": 24, "Longitud_cm": 20.2, "pH": 6.5, "Temperatura": 22.8},
        {"Tiempo_dias": 27, "Longitud_cm": 23.5, "pH": 6.6, "Temperatura": 22.5},
    ]
    
    # Crear modelo
    modelo_lechugas = ModeloCrecimiento('lechuga')
    
    # Cargar datos
    modelo_lechugas.cargar_datos(datos_lechugas)
    
    # Ajustar modelo de Von Bertalanffy
    modelo_lechugas.ajustar_modelo_von_bertalanffy()
    
    # Entrenar modelo lineal para factores ambientales
    modelo_lechugas.entrenar_modelo_lineal()
    
    # Predecir crecimiento para los próximos 15 días
    parametros_futuros = {
        "pH": 6.5,
        "Temperatura": 23.0
    }
    
    predicciones_lechugas = modelo_lechugas.predecir_crecimiento(15, parametros_futuros, 'suavizada')
    
    # Visualizar resultados
    modelo_lechugas.visualizar_resultados(predicciones_lechugas)

# Ejemplo de uso como API
def ejemplo_api():
    print("\n=== EJEMPLO DE USO COMO API ===")
    
    # Datos de ejemplo para peces
    datos_peces = [
        {"Tiempo_dias": 0, "Longitud_cm": 5.0, "pH": 7.2, "Temperatura": 18, "Oxigeno": 6.5},
        {"Tiempo_dias": 7, "Longitud_cm": 5.8, "pH": 7.1, "Temperatura": 18.5, "Oxigeno": 6.3},
        {"Tiempo_dias": 14, "Longitud_cm": 6.7, "pH": 7.3, "Temperatura": 19, "Oxigeno": 6.4},
        {"Tiempo_dias": 21, "Longitud_cm": 7.9, "pH": 7.0, "Temperatura": 19.2, "Oxigeno": 6.2},
        {"Tiempo_dias": 28, "Longitud_cm": 9.2, "pH": 7.2, "Temperatura": 18.8, "Oxigeno": 6.5},
        {"Tiempo_dias": 35, "Longitud_cm": 10.6, "pH": 7.4, "Temperatura": 19.5, "Oxigeno": 6.6},
        {"Tiempo_dias": 42, "Longitud_cm": 12.1, "pH": 7.3, "Temperatura": 19.8, "Oxigeno": 6.4},
    ]
    
    # Parámetros para la API
    tipo_organismo = 'pez'
    dias_predecir = 20
    parametros_futuros = {
        "pH": 7.2,
        "Temperatura": 19.5,
        "Oxigeno": 6.5
    }
    tipo_restriccion = 'historica'
    
    # Llamar a la función de predicción
    resultado = predecir_crecimiento(
        tipo_organismo,
        datos_peces,
        dias_predecir,
        parametros_futuros,
        tipo_restriccion
    )
    
    # Mostrar resultado
    print(f"Tipo de organismo: {resultado['tipo_organismo']}")
    
    if 'error' in resultado:
        print(f"Error: {resultado['error']}")
    else:
        print(f"Parámetros Von Bertalanffy: L_inf={resultado['parametros_vb']['L_inf']:.2f}, K={resultado['parametros_vb']['K']:.4f}, t0={resultado['parametros_vb']['t0']:.2f}")
        print(f"Métricas Von Bertalanffy: R²={resultado['metricas_vb']['r2']:.4f}, RMSE={resultado['metricas_vb']['rmse']:.4f}")
        
        print("\nPredicciones:")
        print(f"{'Día':<8}{'Longitud (cm)':<15}{'Crecimiento diario (cm)':<25}")
        print("-" * 48)
        
        for i, pred in enumerate(resultado["predicciones"]):
            if i % 5 == 0 or i == len(resultado["predicciones"]) - 1:
                print(f"{pred['dia']:<8}{pred['longitud']:<15.4f}{pred['tasa_crecimiento']:<25.6f}")
        
        ultimo_pred = resultado["predicciones"][-1]
        print(f"\nEn el día {ultimo_pred['dia']}, la longitud predicha será {ultimo_pred['longitud']:.4f} cm")

# Ejecutar ejemplos
if __name__ == "__main__":
    ejemplo_peces()
    ejemplo_lechugas()
    ejemplo_api()