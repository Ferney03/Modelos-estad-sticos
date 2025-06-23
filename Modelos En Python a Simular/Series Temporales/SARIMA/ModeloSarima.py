import warnings
import pandas as pd
import matplotlib
# Fix matplotlib backend issue on Windows
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Suprimir advertencias
warnings.filterwarnings("ignore")

class PredictorPeces:
    def __init__(self):
        self.serie_datos = None
        self.modelo = None
        self.modelo_statsmodels = None  # Para acceder a los p-valores
        self.LIMITE_BIOLOGICO = 70  # Límite biológico en cm para truchas
    
    def cargar_datos(self, archivo):
        """Carga y limpia los datos desde un archivo Excel"""
        try:
            df = pd.read_excel(archivo, engine='openpyxl')
            df['Longitud_cm'] = pd.to_numeric(df['Longitud_cm'].astype(str).str.replace(',', '.'), errors='coerce')
            df = df.dropna(subset=['Longitud_cm'])
            
            # Eliminar outliers y valores incoherentes mayores al límite biológico
            df = df[df['Longitud_cm'] <= self.LIMITE_BIOLOGICO]
            
            self.visualizar_datos(df['Longitud_cm'], "Serie Temporal: Longitud de Peces (Datos Limpios)")
            
            self.serie_datos = df['Longitud_cm']
            return True
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return False
    
    def visualizar_datos(self, serie, titulo, datos_reales=None, predicciones=None):
        """Visualiza datos en un gráfico"""
        plt.figure(figsize=(12, 6))
        
        if datos_reales is not None and predicciones is not None:
            plt.plot(datos_reales.index, datos_reales, label="Datos Reales", color='blue', marker='o', alpha=0.7)
            plt.plot(predicciones.index, predicciones, label="Predicciones", color='red', marker='x', alpha=0.7)
            plt.legend()
            # Añadir línea horizontal para el límite biológico
            plt.axhline(y=self.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                       label=f"Límite biológico ({self.LIMITE_BIOLOGICO} cm)")
            plt.legend()
        else:
            plt.plot(serie, marker='o', linestyle='-', alpha=0.7)
            # Añadir línea horizontal para el límite biológico
            plt.axhline(y=self.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                       label=f"Límite biológico ({self.LIMITE_BIOLOGICO} cm)")
            plt.legend()
        
        plt.title(titulo)
        plt.xlabel("Índice")
        plt.ylabel("Longitud (cm)")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def mostrar_graficas_autocorrelacion(self, serie, max_lags=40):
        """Muestra gráficas de autocorrelación (ACF) y autocorrelación parcial (PACF)"""
        # Crear una figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico ACF (Autocorrelation Function)
        plot_acf(serie, lags=max_lags, ax=ax1, alpha=0.05)
        ax1.set_title('Función de Autocorrelación (ACF)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico PACF (Partial Autocorrelation Function)
        plot_pacf(serie, lags=max_lags, ax=ax2, alpha=0.05)
        ax2.set_title('Función de Autocorrelación Parcial (PACF)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        
        return fig
    
    def mostrar_graficas_autocorrelacion_diferenciada(self, serie, d=1, max_lags=40):
        """Muestra gráficas de autocorrelación para la serie diferenciada"""
        # Aplicar diferenciación
        serie_diff = serie.diff(d).dropna()
        
        # Crear una figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico ACF (Autocorrelation Function) para la serie diferenciada
        plot_acf(serie_diff, lags=max_lags, ax=ax1, alpha=0.05)
        ax1.set_title(f'Función de Autocorrelación (ACF) - Serie Diferenciada d={d}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico PACF (Partial Autocorrelation Function) para la serie diferenciada
        plot_pacf(serie_diff, lags=max_lags, ax=ax2, alpha=0.05)
        ax2.set_title(f'Función de Autocorrelación Parcial (PACF) - Serie Diferenciada d={d}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        
        return fig
    
    def verificar_estacionariedad(self, serie):
        """Verifica la estacionariedad de la serie y devuelve la serie diferenciada y el orden de diferenciación"""
        dftest = adfuller(serie, autolag='AIC')
        pvalue = dftest[1]
        
        # Mostrar el p-valor directamente en la consola
        print("\n============= ANÁLISIS DE ESTACIONARIEDAD =============")
        print(f"Test Dickey-Fuller aumentado: P-VALOR = {pvalue:.6f}")
        
        if pvalue < 0.05:
            print("La serie es estacionaria (p < 0.05)")
            # Mostrar gráficas de autocorrelación para la serie estacionaria original
            self.mostrar_graficas_autocorrelacion(serie)
            return serie, 0
        
        ts_diff = serie.diff().dropna()
        dftest = adfuller(ts_diff, autolag='AIC')
        pvalue = dftest[1]
        
        print(f"Test en primera diferencia: P-VALOR = {pvalue:.6f}")
        
        if pvalue < 0.05:
            print("La serie es estacionaria en primera diferencia (p < 0.05)")
            # Mostrar gráficas de autocorrelación para la serie diferenciada una vez
            self.mostrar_graficas_autocorrelacion_diferenciada(serie, d=1)
            return ts_diff, 1
        
        ts_diff2 = ts_diff.diff().dropna()
        dftest = adfuller(ts_diff2, autolag='AIC')
        pvalue = dftest[1]
        
        print(f"Test en segunda diferencia: P-VALOR = {pvalue:.6f}")
        print("La serie es estacionaria en segunda diferencia (p < 0.05)" if pvalue < 0.05 else "¡Advertencia! La serie podría no ser estacionaria incluso después de la segunda diferenciación.")
        print("===========================================================\n")
        
        # Mostrar gráficas de autocorrelación para la serie diferenciada dos veces
        self.mostrar_graficas_autocorrelacion_diferenciada(serie, d=2)
        
        return ts_diff2, 2
    
    def encontrar_mejor_modelo(self, serie, d_recomendado=None):
        """Encuentra el mejor modelo ARIMA para la serie temporal"""
        seasonal = len(serie) >= 24 and abs(serie.autocorr(lag=12)) > 0.3
        m = 12 if seasonal else 1
        
        print("\n========== BÚSQUEDA DEL MEJOR MODELO SARIMA ==========")
        self.modelo = auto_arima(
            serie,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=d_recomendado,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            information_criterion='aic',
            n_fits=50
        )
        
        # Después de encontrar el mejor modelo con auto_arima, creamos el modelo equivalente
        # en statsmodels para acceder a los p-valores
        order = self.modelo.order
        seasonal_order = self.modelo.seasonal_order
        
        print(f"\nAjustando modelo SARIMA{order}{seasonal_order} en statsmodels para obtener p-valores...")
        
        # Ajustar el modelo con statsmodels para obtener p-valores
        try:
            self.modelo_statsmodels = SARIMAX(
                serie,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            
            # Imprimir directamente la tabla de parámetros con p-valores
            print("\n=============== P-VALORES DEL MODELO ===============")
            tabla = self.modelo_statsmodels.summary().tables[1].as_text()
            print(tabla)
            print("===========================================================\n")
            
        except Exception as e:
            print(f"Error al ajustar modelo statsmodels: {e}")
            self.modelo_statsmodels = None
        
        return self.modelo
    
    def analizar_residuos(self):
        """Analiza los residuos del modelo para verificar su calidad"""
        if self.modelo_statsmodels is None:
            raise ValueError("No se ha ajustado ningún modelo statsmodels.")
        
        # Obtener residuos
        residuos = self.modelo_statsmodels.resid
        
        # Crear figura para los residuos
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gráfico de residuos
        axes[0, 0].plot(residuos, marker='o', linestyle='', alpha=0.7)
        axes[0, 0].set_title('Residuos del Modelo')
        axes[0, 0].set_xlabel('Índice')
        axes[0, 0].set_ylabel('Residuo')
        axes[0, 0].axhline(y=0, color='r', linestyle='-')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histograma de residuos
        axes[0, 1].hist(residuos, bins=25, alpha=0.7, density=True)
        axes[0, 1].set_title('Histograma de Residuos')
        axes[0, 1].set_xlabel('Residuo')
        axes[0, 1].set_ylabel('Densidad')
        
        # Añadir curva normal para comparar
        import scipy.stats as stats
        x = np.linspace(min(residuos), max(residuos), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, np.mean(residuos), np.std(residuos)),
                       color='red', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF de residuos
        plot_acf(residuos, lags=40, ax=axes[1, 0], alpha=0.05)
        axes[1, 0].set_title('ACF de Residuos')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuos, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot Residuos')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def obtener_pvalores(self):
        """Obtiene los p-valores de los parámetros del modelo"""
        if self.modelo_statsmodels is None:
            return "No se ha ajustado ningún modelo con statsmodels. No se pueden obtener p-valores."
        
        try:
            # Extraer tabla de p-valores
            resultados = self.modelo_statsmodels.summary()
            tabla_pvalores = resultados.tables[1]
            
            # Extraer los p-valores como valores numéricos para poder trabajar con ellos
            parametros_significativos = 0
            total_parametros = 0
            
            # Convertir la tabla a texto para facilitar la extracción
            tabla_texto = tabla_pvalores.as_text()
            
            # Crear una versión formateada para la visualización
            lineas = tabla_texto.split('\n')
            tabla_formateada = "P-VALORES DE LOS PARÁMETROS:\n"
            
            for linea in lineas:
                if 'P>|z|' in linea or '===' in linea or 'coef' in linea:
                    tabla_formateada += linea + "\n"
                elif len(linea.strip()) > 0 and not linea.strip().startswith("=="):
                    # Es una línea con un parámetro
                    partes = linea.split()
                    if len(partes) >= 4:  # Asegurarse de que hay suficientes columnas
                        param_name = partes[0]
                        p_valor = float(partes[4]) if partes[4] != 'nan' else 1.0
                        
                        # Contar parámetros significativos
                        total_parametros += 1
                        if p_valor < 0.05:
                            parametros_significativos += 1
                            tabla_formateada += f"{linea} *** SIGNIFICATIVO ***\n"
                        else:
                            tabla_formateada += f"{linea} (no significativo)\n"
            
            # Agregar resumen al final
            if total_parametros > 0:
                tabla_formateada += f"\nResumen: {parametros_significativos} de {total_parametros} "
                tabla_formateada += f"parámetros son estadísticamente significativos (p < 0.05).\n"
            
            return tabla_formateada
            
        except Exception as e:
            return f"Error al obtener p-valores: {e}"
    
    def evaluar_prediccion(self, datos_reales, predicciones):
        """Evalúa la calidad de las predicciones con múltiples métricas"""
        # Error Cuadrático Medio (MSE)
        mse = mean_squared_error(datos_reales, predicciones)
        
        # Raíz del Error Cuadrático Medio (RMSE)
        rmse = np.sqrt(mse)
        
        # Error Absoluto Medio (MAE)
        try:
            mae = np.mean(np.abs(datos_reales - predicciones))
        except:
            mae = np.nan  # En caso de error, asignar NaN
        
        # R Cuadrado (R²)
        try:
            r2 = r2_score(datos_reales, predicciones)
        except:
            r2 = 0.0  # En caso de error, asignar 0
        
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.visualizar_datos(None, "Comparación de Datos Reales vs. Predicciones", 
                             datos_reales, predicciones)
        
        return mse, rmse, mae, r2
    
    def predecir_longitud(self, dias_a_predecir):
        """Realiza la predicción de longitud para los días especificados"""
        if self.serie_datos is None:
            raise ValueError("Debe cargar datos primero.")
        
        serie = self.serie_datos.dropna().astype(float)
        
        if len(serie) < 10:
            raise ValueError("La serie de tiempo tiene muy pocos datos para el modelo (mínimo 10).")
        
        serie_diff, d_recomendado = self.verificar_estacionariedad(serie)
        self.encontrar_mejor_modelo(serie, d_recomendado)
        
        # Mostrar información sobre el modelo seleccionado
        print(f"\n============= RESUMEN DEL MODELO SELECCIONADO =============")
        print(f"Modelo seleccionado: {self.modelo}")
        print(f"Orden SARIMA: {self.modelo.order} (p,d,q), Estacional: {self.modelo.seasonal_order} (P,D,Q,s)")
        
        # Imprimir información de p-valores
        print("\nP-valores de los parámetros del modelo:")
        p_valores_info = self.obtener_pvalores()
        print(p_valores_info)
        print("===========================================================\n")
        
        # Analizar residuos del modelo
        print("\nAnalizando residuos del modelo...")
        self.analizar_residuos()
        
        # Generar predicciones
        predicciones, conf_int = self.modelo.predict(n_periods=dias_a_predecir, return_conf_int=True, alpha=0.05)
        
        # Crear un índice para las predicciones
        indices_pred = pd.RangeIndex(start=len(serie), stop=len(serie) + dias_a_predecir)
        predicciones = pd.Series(predicciones, index=indices_pred)
        
        # Calcular el crecimiento promedio diario en los datos históricos (usando los últimos 30 días o todos si hay menos)
        n_historico = min(30, len(serie))
        ultimo_valor = serie.iloc[-1]
        
        # Aplicar límites biológicos y asegurar crecimiento coherente
        for i in range(len(predicciones)):
            # Si la predicción excede el límite biológico, ajustarla
            if predicciones.iloc[i] > self.LIMITE_BIOLOGICO:
                predicciones.iloc[i] = self.LIMITE_BIOLOGICO
            
            # Si hay decrecimiento (lo cual no es biológicamente coherente para truchas jóvenes), 
            # mantener el valor anterior o aplicar un crecimiento mínimo
            if i > 0 and predicciones.iloc[i] < predicciones.iloc[i-1]:
                # Aplicar un pequeño crecimiento (max 0.1% del valor actual)
                predicciones.iloc[i] = predicciones.iloc[i-1] + min(0.001 * predicciones.iloc[i-1], 0.05)
            
            # Si estamos acercándonos al límite biológico, desacelerar el crecimiento
            if predicciones.iloc[i] > self.LIMITE_BIOLOGICO * 0.95:
                if i > 0:
                    # Calcular crecimiento logarítmico a medida que se acerca al límite
                    distancia_al_limite = self.LIMITE_BIOLOGICO - predicciones.iloc[i-1]
                    crecimiento = distancia_al_limite * 0.05  # 5% de la distancia restante al límite
                    predicciones.iloc[i] = predicciones.iloc[i-1] + crecimiento
        
        return predicciones

# Interfaz gráfica mejorada
class InterfazPredictor:
    def __init__(self, root):
        self.root = root
        self.predictor = PredictorPeces()
        self.configurar_interfaz()
    
    def configurar_interfaz(self):
        """Configura los elementos de la interfaz gráfica"""
        self.root.title("Predicción Adaptable de Longitud de Peces")
        self.root.geometry("800x780")  # Aumentado para mostrar más información
        self.root.configure(bg="#f0f4f8")
        
        main_frame = tk.Frame(self.root, bg="#f0f4f8", padx=25, pady=25)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        titulo = tk.Label(main_frame, text="Sistema de Predicción de Crecimiento de Truchas", 
                         font=("Arial", 18, "bold"), bg="#f0f4f8", fg="#1a5276")
        titulo.pack(pady=15)
        
        # Subtítulo con información del límite biológico
        subtitulo = tk.Label(main_frame, 
                           text=f"(Respetando el límite biológico de {self.predictor.LIMITE_BIOLOGICO} cm)", 
                           font=("Arial", 12), bg="#f0f4f8", fg="#2874a6")
        subtitulo.pack()
        
        # Marco para entrada de datos
        datos_frame = tk.Frame(main_frame, bg="#e1e8ed", padx=15, pady=15, bd=2, relief=tk.GROOVE)
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
        
        # Botones de análisis
        btns_frame = tk.Frame(main_frame, bg="#f0f4f8", pady=10)
        btns_frame.pack(fill=tk.X)
        
        self.btn_predecir = tk.Button(btns_frame, text="Realizar Predicción", command=self.realizar_prediccion, 
                                 font=("Arial", 14, "bold"), bg="#2ecc71", fg="white", padx=20, pady=10)
        self.btn_predecir.pack(side=tk.LEFT, padx=10)
        
        self.btn_autocorr = tk.Button(btns_frame, text="Ver Autocorrelación", command=self.mostrar_autocorrelacion, 
                                 font=("Arial", 14), bg="#9b59b6", fg="white", padx=20, pady=10)
        self.btn_autocorr.pack(side=tk.LEFT, padx=10)
        
        # Sección para p-valores del modelo
        estadisticas_frame = tk.Frame(main_frame, bg="#e5e8e8", padx=15, pady=15, bd=2, relief=tk.GROOVE)
        estadisticas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(estadisticas_frame, text="Estadísticas y P-Valores del Modelo", font=("Arial", 14, "bold"), 
               bg="#e5e8e8", fg="#34495e").pack(pady=5)
        
        # Área para mostrar estadísticas y p-valores
        self.estadisticas_text = tk.Text(estadisticas_frame, font=("Consolas", 10), height=15, width=65)
        self.estadisticas_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Marco para resultados de predicción
        resultados_frame = tk.Frame(main_frame, bg="#eaecee", padx=15, pady=15, bd=2, relief=tk.GROOVE)
        resultados_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(resultados_frame, text="Resultados de Predicción", font=("Arial", 14, "bold"), 
               bg="#eaecee", fg="#34495e").pack(pady=5)
        
        # Área para mostrar resultados
        self.resultado_text = tk.Text(resultados_frame, font=("Consolas", 11), height=10, width=65)
        self.resultado_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Barra de estado
        self.status_bar = tk.Label(self.root, text="Listo", bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                              font=("Arial", 10), bg="#d6eaf8", fg="#2c3e50")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def cargar_archivo(self):
        """Maneja la carga de archivo"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de datos", 
            filetypes=[("Archivos Excel", "*.xlsx *.xls")]
        )
        
        if not archivo:
            return
        
        self.archivo_path = archivo
        self.lbl_archivo.config(text=f"Archivo: {archivo.split('/')[-1]}")
        self.status_bar.config(text="Archivo cargado exitosamente")
    
    def mostrar_autocorrelacion(self):
        """Muestra las gráficas de autocorrelación para los datos cargados"""
        if not hasattr(self, 'archivo_path'):
            messagebox.showerror("Error", "Debe seleccionar un archivo primero.")
            return
        
        try:
            # Cargar datos si no se han cargado ya
            if self.predictor.serie_datos is None:
                if not self.predictor.cargar_datos(self.archivo_path):
                    messagebox.showerror("Error", "No se pudieron cargar los datos correctamente.")
                    return
            
            serie = self.predictor.serie_datos.dropna().astype(float)
            
            # Mostrar ventana informativa sobre los gráficos
            messagebox.showinfo("Análisis de Autocorrelación", 
                             "Se mostrarán tres gráficos:\n\n"
                             "1. ACF y PACF de la serie original\n"
                             "2. ACF y PACF de la serie diferenciada (si es necesario)\n"
                             "3. Análisis de residuos (si hay un modelo ajustado)\n\n"
                             "Los gráficos de ACF y PACF ayudan a determinar los órdenes p, q, P, Q del modelo SARIMA.")
            
            # Verificar estacionariedad y mostrar gráficos de autocorrelación
            self.predictor.verificar_estacionariedad(serie)
            
            # Si hay un modelo ajustado, mostrar análisis de residuos
            if self.predictor.modelo_statsmodels is not None:
                self.predictor.analizar_residuos()
            
            self.status_bar.config(text="Análisis de autocorrelación completado")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error detallado: {error_details}")
            messagebox.showerror("Error", f"Ocurrió un error durante el análisis: {str(e)}")
            self.status_bar.config(text="Error en el análisis de autocorrelación")
    
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
            
            # Cargar datos
            if not self.predictor.cargar_datos(self.archivo_path):
                messagebox.showerror("Error", "No se pudieron cargar los datos correctamente.")
                return
            
            # Dividir datos para entrenamiento y prueba
            self.status_bar.config(text="Realizando predicción...")
            self.root.update()
            
            serie = self.predictor.serie_datos
            train_size = int(len(serie) * 0.8)
            train, test = serie[:train_size], serie[train_size:]
            
            # Crear una variable para capturar la salida de la consola
            import io
            import sys
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Realizar predicción
            predicciones = self.predictor.predecir_longitud(dias)
            
            # Restaurar stdout y obtener la salida
            sys.stdout = old_stdout
            output = new_stdout.getvalue()
            
            # Guardar a un archivo para depuración
            with open("debug_output.txt", "w") as f:
                f.write(output)
            
            # Mostrar información estadística y p-valores
            self.mostrar_estadisticas(output)
            
            # Evaluar si hay suficientes datos de prueba
            test_compare = test[:min(dias, len(test))]
            if len(test_compare) > 0:
                mse, rmse, mae, r2 = self.predictor.evaluar_prediccion(test_compare, predicciones[:len(test_compare)])
                self.mostrar_resultados(predicciones, mse, rmse, mae, r2)
            else:
                self.mostrar_resultados(predicciones)
            
            # Informar al usuario sobre los gráficos de autocorrelación
            messagebox.showinfo("Análisis Completado", 
                             "La predicción se ha completado con éxito.\n\n"
                             "Se han generado gráficos de autocorrelación durante el análisis.\n"
                             "Puede volver a verlos usando el botón 'Ver Autocorrelación'.")
            
            self.status_bar.config(text="Predicción completada respetando límites biológicos. Ver P-valores en pestaña superior.")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error detallado: {error_details}")
            messagebox.showerror("Error", f"Ocurrió un error durante la predicción: {str(e)}")
            self.status_bar.config(text="Error en la predicción")
    
    def mostrar_estadisticas(self, output):
        """Muestra la información estadística y p-valores en el área de texto correspondiente"""
        self.estadisticas_text.delete(1.0, tk.END)
        
        # Información de encabezado
        self.estadisticas_text.insert(tk.END, "ANÁLISIS ESTADÍSTICO DEL MODELO SARIMA\n")
        self.estadisticas_text.insert(tk.END, "==========================================\n\n")
        
        # Extraer información relevante de la salida capturada
        p_valores_encontrados = False
        
        # Buscar secciones específicas en la salida
        lines = output.split('\n')
        
        # Analizar las líneas buscando secciones de p-valores
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Buscar sección de estacionariedad
            if "ANÁLISIS DE ESTACIONARIEDAD" in line:
                self.estadisticas_text.insert(tk.END, "ANÁLISIS DE ESTACIONARIEDAD:\n")
                i += 1
                # Extraer las próximas líneas hasta encontrar la sección de separación
                while i < len(lines) and "=====" not in lines[i]:
                    if "P-VALOR" in lines[i]:
                        self.estadisticas_text.insert(tk.END, lines[i] + "\n")
                        p_valores_encontrados = True
                    i += 1
                self.estadisticas_text.insert(tk.END, "\n")
            
            # Buscar la sección de p-valores del modelo
            elif "P-VALORES DEL MODELO" in line:
                self.estadisticas_text.insert(tk.END, "P-VALORES DEL MODELO SARIMA:\n")
                self.estadisticas_text.insert(tk.END, "-------------------------------------------\n")
                i += 1
                # Extraer las líneas de la tabla hasta encontrar la sección de separación
                while i < len(lines) and "=====" not in lines[i]:
                    self.estadisticas_text.insert(tk.END, lines[i] + "\n")
                    if "P>|z|" in lines[i] or "pvalue" in lines[i].lower():
                        p_valores_encontrados = True
                    i += 1
                self.estadisticas_text.insert(tk.END, "\n")
            
            # Buscar el resumen del modelo seleccionado
            elif "RESUMEN DEL MODELO SELECCIONADO" in line:
                self.estadisticas_text.insert(tk.END, "INFORMACIÓN DEL MODELO:\n")
                i += 1
                # Extraer las líneas sobre el modelo
                while i < len(lines) and "P-valores" not in lines[i]:
                    if "SARIMA" in lines[i] or "Modelo" in lines[i]:
                        self.estadisticas_text.insert(tk.END, lines[i] + "\n")
                    i += 1
                self.estadisticas_text.insert(tk.END, "\n")
            
            i += 1
        
        # Si no se encontraron p-valores, mostrar mensaje informativo
        if not p_valores_encontrados:
            self.estadisticas_text.insert(tk.END, "\nADVERTENCIA: No se pudieron detectar p-valores en la salida.\n")
            self.estadisticas_text.insert(tk.END, "Esto puede deberse a un problema en la ejecución del modelo.\n")
            self.estadisticas_text.insert(tk.END, "Se ha guardado la salida completa en 'debug_output.txt' para diagnóstico.\n\n")
        
        # Añadir interpretación sobre autocorrelación
        self.estadisticas_text.insert(tk.END, "\nACERCA DE LOS GRÁFICOS DE AUTOCORRELACIÓN:\n")
        self.estadisticas_text.insert(tk.END, "- ACF: Muestra relaciones entre observaciones separadas por k periodos\n")
        self.estadisticas_text.insert(tk.END, "- PACF: Muestra correlaciones parciales eliminando efectos intermedios\n")
        self.estadisticas_text.insert(tk.END, "- ACF significativa en rezago q → Orden MA(q)\n")
        self.estadisticas_text.insert(tk.END, "- PACF significativa en rezago p → Orden AR(p)\n\n")
        
        # Añadir interpretación de p-valores
        self.estadisticas_text.insert(tk.END, "\nINTERPRETACIÓN DE P-VALORES:\n")
        self.estadisticas_text.insert(tk.END, "- p < 0.05: El parámetro es estadísticamente significativo\n")
        self.estadisticas_text.insert(tk.END, "- p ≥ 0.05: El parámetro podría no ser necesario en el modelo\n\n")
        self.estadisticas_text.insert(tk.END, "Nota: Un modelo con parámetros no significativos puede indicar sobreajuste.\n")
    
    def mostrar_resultados(self, predicciones, mse=None, rmse=None, mae=None, r2=None):
        """Muestra los resultados en el área de texto"""
        self.resultado_text.delete(1.0, tk.END)
    
        # Mostrar métricas si están disponibles
        if mse is not None:
            self.resultado_text.insert(tk.END, f"Métricas de evaluación:\n")
            self.resultado_text.insert(tk.END, f"Error Cuadrático Medio (MSE): {mse:.4f}\n")
            self.resultado_text.insert(tk.END, f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}\n")
            
            # Manejar posibles valores NaN en MAE
            if np.isnan(mae):
                self.resultado_text.insert(tk.END, f"Error Absoluto Medio (MAE): No disponible\n")
            else:
                self.resultado_text.insert(tk.END, f"Error Absoluto Medio (MAE): {mae:.4f}\n")
            
            self.resultado_text.insert(tk.END, f"R Cuadrado (R²): {r2:.4f}\n\n")
        
        # Mostrar predicciones
        self.resultado_text.insert(tk.END, "Predicciones para los próximos días:\n")
        self.resultado_text.insert(tk.END, f"(Límite biológico: {self.predictor.LIMITE_BIOLOGICO} cm)\n\n")
        
        # Calcular tasas de crecimiento diario
        crecimientos = []
        valores_prediccion = []
        
        # Crear una ventana adicional para mostrar la tabla de crecimiento
        tabla_window = tk.Toplevel(self.root)
        tabla_window.title("Tabla de Crecimiento Diario")
        tabla_window.geometry("600x500")
        tabla_window.configure(bg="#f0f4f8")
        
        # Crear un frame para la tabla
        tabla_frame = tk.Frame(tabla_window, bg="#f0f4f8", padx=20, pady=20)
        tabla_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título de la tabla
        tk.Label(tabla_frame, text="Predicción de Crecimiento Diario", 
                font=("Arial", 16, "bold"), bg="#f0f4f8", fg="#1a5276").pack(pady=10)
        
        # Crear encabezados de la tabla
        encabezados_frame = tk.Frame(tabla_frame, bg="#3498db")
        encabezados_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(encabezados_frame, text="Día", font=("Arial", 12, "bold"), 
                bg="#3498db", fg="white", width=8, padx=5, pady=5).pack(side=tk.LEFT)
        tk.Label(encabezados_frame, text="Longitud (cm)", font=("Arial", 12, "bold"), 
                bg="#3498db", fg="white", width=15, padx=5, pady=5).pack(side=tk.LEFT)
        tk.Label(encabezados_frame, text="Crecimiento (cm)", font=("Arial", 12, "bold"), 
                bg="#3498db", fg="white", width=15, padx=5, pady=5).pack(side=tk.LEFT)
        tk.Label(encabezados_frame, text="Estado", font=("Arial", 12, "bold"), 
                bg="#3498db", fg="white", width=20, padx=5, pady=5).pack(side=tk.LEFT)
        
        # Crear un canvas con scrollbar para la tabla
        canvas = tk.Canvas(tabla_frame, bg="#f0f4f8")
        scrollbar = tk.Scrollbar(tabla_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f4f8")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Procesar y mostrar cada predicción
        for i, valor in enumerate(predicciones.values):
            valores_prediccion.append(valor)
            
            # Calcular crecimiento desde el día anterior
            crecimiento = 0
            if i > 0:
                crecimiento = valor - predicciones.values[i-1]
                crecimientos.append(crecimiento)
            
            # Determinar el estado del crecimiento
            estado = ""
            color_fila = "#ffffff"  # Color por defecto
            
            if i == 0:
                estado = "Valor inicial"
                color_fila = "#e8f4f8"
            elif valor >= self.predictor.LIMITE_BIOLOGICO * 0.95:
                estado = "¡Cerca del límite biológico!"
                color_fila = "#f8d7da"  # Rojo claro
            elif crecimiento > 0.1:
                estado = "Crecimiento rápido"
                color_fila = "#d4edda"  # Verde claro
            elif crecimiento < 0.01:
                estado = "Crecimiento lento"
                color_fila = "#fff3cd"  # Amarillo claro
            else:
                estado = "Crecimiento normal"
                color_fila = "#e8f4f8"  # Azul claro
            
            # Crear fila en la tabla
            fila_frame = tk.Frame(scrollable_frame, bg=color_fila)
            fila_frame.pack(fill=tk.X, pady=1)
            
            tk.Label(fila_frame, text=f"{i+1}", font=("Arial", 11), 
                    bg=color_fila, width=8, padx=5, pady=3).pack(side=tk.LEFT)
            tk.Label(fila_frame, text=f"{valor:.2f}", font=("Arial", 11), 
                    bg=color_fila, width=15, padx=5, pady=3).pack(side=tk.LEFT)
            
            if i > 0:
                tk.Label(fila_frame, text=f"+{crecimiento:.4f}", font=("Arial", 11), 
                        bg=color_fila, width=15, padx=5, pady=3).pack(side=tk.LEFT)
            else:
                tk.Label(fila_frame, text="-", font=("Arial", 11), 
                        bg=color_fila, width=15, padx=5, pady=3).pack(side=tk.LEFT)
            
            tk.Label(fila_frame, text=estado, font=("Arial", 11), 
                    bg=color_fila, width=20, padx=5, pady=3).pack(side=tk.LEFT)
            
            # Mostrar en el área de texto principal (versión resumida)
            if valor >= self.predictor.LIMITE_BIOLOGICO * 0.95:
                self.resultado_text.insert(tk.END, f"Día {i+1}: {valor:.2f} cm (¡Cerca del límite biológico!)\n")
            else:
                self.resultado_text.insert(tk.END, f"Día {i+1}: {valor:.2f} cm\n")
            
            if i > 0:
                self.resultado_text.insert(tk.END, f"   Crecimiento: +{crecimiento:.4f} cm\n")
        
        # Añadir botones para visualizar gráficos
        botones_frame = tk.Frame(tabla_window, bg="#f0f4f8", pady=10)
        botones_frame.pack(fill=tk.X)
        
        btn_grafico_longitud = tk.Button(botones_frame, text="Ver Gráfico de Longitud", 
                                     command=lambda: self.mostrar_grafico_longitud(valores_prediccion),
                                     font=("Arial", 12), bg="#3498db", fg="white", padx=10, pady=5)
        btn_grafico_longitud.pack(side=tk.LEFT, padx=10)
        
        btn_grafico_crecimiento = tk.Button(botones_frame, text="Ver Gráfico de Crecimiento", 
                                       command=lambda: self.mostrar_grafico_crecimiento(crecimientos),
                                       font=("Arial", 12), bg="#2ecc71", fg="white", padx=10, pady=5)
        btn_grafico_crecimiento.pack(side=tk.LEFT, padx=10)
        
        # Crear y mostrar gráfico de crecimiento diario automáticamente
        if len(crecimientos) > 0:
            self.mostrar_grafico_crecimiento(crecimientos)
    
    def mostrar_grafico_longitud(self, valores):
        """Muestra un gráfico de la longitud predicha"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(valores)+1), valores, marker='o', linestyle='-', color='blue', alpha=0.7)
        plt.axhline(y=self.predictor.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                   label=f"Límite biológico ({self.predictor.LIMITE_BIOLOGICO} cm)")
        plt.title('Predicción de Longitud por Día')
        plt.xlabel('Día')
        plt.ylabel('Longitud (cm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def mostrar_grafico_crecimiento(self, crecimientos):
        """Muestra un gráfico del crecimiento diario predicho"""
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(crecimientos)+1), crecimientos, color='green', alpha=0.7)
        plt.title('Tasa de Crecimiento Diario Predicho')
        plt.xlabel('Día')
        plt.ylabel('Crecimiento (cm/día)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Punto de entrada principal
if __name__ == "__main__":
    # Imprimir un mensaje al iniciar para comprobar que está funcionando
    print("Iniciando Sistema de Predicción SARIMA con Análisis de Autocorrelación...")
    
    root = tk.Tk()
    app = InterfazPredictor(root)
    root.mainloop()