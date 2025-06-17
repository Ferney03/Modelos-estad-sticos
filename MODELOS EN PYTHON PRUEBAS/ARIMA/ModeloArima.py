import warnings
import pandas as pd
import matplotlib
# Fix matplotlib backend issue on Windows
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import io
import sys
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

class PredictorPeces:
    def __init__(self):
        self.serie_datos = None
        self.modelo = None
        self.modelo_statsmodels = None
        self.LIMITE_BIOLOGICO = 70  # Biological limit in cm for trout
    
    def cargar_datos(self, archivo):
        """Load and clean data from Excel file"""
        try:
            df = pd.read_excel(archivo, engine='openpyxl')
            df['Longitud_cm'] = pd.to_numeric(df['Longitud_cm'].astype(str).str.replace(',', '.'), errors='coerce')
            df = df.dropna(subset=['Longitud_cm'])
            
            # Remove outliers and incoherent values greater than biological limit
            df = df[df['Longitud_cm'] <= self.LIMITE_BIOLOGICO]
            
            self.visualizar_datos(df['Longitud_cm'], "Serie Temporal: Longitud de Peces (Datos Limpios)")
            
            self.serie_datos = df['Longitud_cm']
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def visualizar_datos(self, serie, titulo, datos_reales=None, predicciones=None):
        """Visualize data in a plot"""
        try:
            plt.figure(figsize=(12, 6))
            
            if datos_reales is not None and predicciones is not None:
                plt.plot(datos_reales.index, datos_reales, label="Datos Reales", color='blue', marker='o', alpha=0.7)
                plt.plot(predicciones.index, predicciones, label="Predicciones", color='red', marker='x', alpha=0.7)
                plt.legend()
                plt.axhline(y=self.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                           label=f"Límite biológico ({self.LIMITE_BIOLOGICO} cm)")
                plt.legend()
            else:
                plt.plot(serie, marker='o', linestyle='-', alpha=0.7)
                plt.axhline(y=self.LIMITE_BIOLOGICO, color='r', linestyle='--', alpha=0.5, 
                           label=f"Límite biológico ({self.LIMITE_BIOLOGICO} cm)")
                plt.legend()
            
            plt.title(titulo)
            plt.xlabel("Índice")
            plt.ylabel("Longitud (cm)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def mostrar_graficas_autocorrelacion(self, serie, max_lags=40):
        """Show autocorrelation (ACF) and partial autocorrelation (PACF) plots"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # ACF plot
            plot_acf(serie, lags=max_lags, ax=ax1, alpha=0.05)
            ax1.set_title('Función de Autocorrelación (ACF)', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # PACF plot
            plot_pacf(serie, lags=max_lags, ax=ax2, alpha=0.05)
            ax2.set_title('Función de Autocorrelación Parcial (PACF)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)
            plt.show()
            
            return fig
        except Exception as e:
            print(f"Error in autocorrelation plots: {e}")
            return None
    
    def mostrar_graficas_autocorrelacion_diferenciada(self, serie, d=1, max_lags=40):
        """Show autocorrelation plots for differenced series"""
        try:
            serie_diff = serie.diff(d).dropna()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            plot_acf(serie_diff, lags=max_lags, ax=ax1, alpha=0.05)
            ax1.set_title(f'Función de Autocorrelación (ACF) - Serie Diferenciada d={d}', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            plot_pacf(serie_diff, lags=max_lags, ax=ax2, alpha=0.05)
            ax2.set_title(f'Función de Autocorrelación Parcial (PACF) - Serie Diferenciada d={d}', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)
            plt.show()
            
            return fig
        except Exception as e:
            print(f"Error in differenced autocorrelation plots: {e}")
            return None
    
    def analizar_residuos(self):
        """Analyze model residuals to verify quality"""
        if self.modelo_statsmodels is None:
            raise ValueError("No statsmodels model has been fitted.")
        
        try:
            residuos = self.modelo_statsmodels.resid
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Residuals plot
            axes[0, 0].plot(residuos, marker='o', linestyle='', alpha=0.7)
            axes[0, 0].set_title('Residuos del Modelo')
            axes[0, 0].set_xlabel('Índice')
            axes[0, 0].set_ylabel('Residuo')
            axes[0, 0].axhline(y=0, color='r', linestyle='-')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[0, 1].hist(residuos, bins=25, alpha=0.7, density=True)
            axes[0, 1].set_title('Histograma de Residuos')
            axes[0, 1].set_xlabel('Residuo')
            axes[0, 1].set_ylabel('Densidad')
            
            # Add normal curve for comparison
            x = np.linspace(min(residuos), max(residuos), 100)
            axes[0, 1].plot(x, stats.norm.pdf(x, np.mean(residuos), np.std(residuos)),
                           color='red', alpha=0.7)
            axes[0, 1].grid(True, alpha=0.3)
            
            # ACF of residuals
            plot_acf(residuos, lags=40, ax=axes[1, 0], alpha=0.05)
            axes[1, 0].set_title('ACF de Residuos')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuos, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot Residuos')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return fig
        except Exception as e:
            print(f"Error in residual analysis: {e}")
            return None
    
    def verificar_estacionariedad(self, serie):
        """Verify stationarity of series and return differenced series and order"""
        dftest = adfuller(serie, autolag='AIC')
        pvalue = dftest[1]
        
        print("\n============= ANÁLISIS DE ESTACIONARIEDAD =============")
        print(f"Test Dickey-Fuller aumentado: P-VALOR = {pvalue:.6f}")
        
        if pvalue < 0.05:
            print("La serie es estacionaria (p < 0.05)")
            self.mostrar_graficas_autocorrelacion(serie)
            return serie, 0
        
        ts_diff = serie.diff().dropna()
        dftest = adfuller(ts_diff, autolag='AIC')
        pvalue = dftest[1]
        
        print(f"Test en primera diferencia: P-VALOR = {pvalue:.6f}")
        
        if pvalue < 0.05:
            print("La serie es estacionaria en primera diferencia (p < 0.05)")
            self.mostrar_graficas_autocorrelacion_diferenciada(serie, d=1)
            return ts_diff, 1
        
        ts_diff2 = ts_diff.diff().dropna()
        dftest = adfuller(ts_diff2, autolag='AIC')
        pvalue = dftest[1]
        
        print(f"Test en segunda diferencia: P-VALOR = {pvalue:.6f}")
        print("La serie es estacionaria en segunda diferencia (p < 0.05)" if pvalue < 0.05 else "¡Advertencia! La serie podría no ser estacionaria incluso después de la segunda diferenciación.")
        print("===========================================================\n")
        
        self.mostrar_graficas_autocorrelacion_diferenciada(serie, d=2)
        
        return ts_diff2, 2
    
    def encontrar_mejor_modelo(self, serie, d_recomendado=None):
        """Find the best ARIMA model for the time series"""
        print("\n========== BÚSQUEDA DEL MEJOR MODELO ARIMA ==========")
        self.modelo = auto_arima(
            serie,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=d_recomendado,
            seasonal=False,
            stepwise=True,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            information_criterion='aic',
            n_fits=50
        )
        
        order = self.modelo.order
        
        print(f"\nAjustando modelo ARIMA{order} en statsmodels para obtener p-valores...")
        
        try:
            self.modelo_statsmodels = ARIMA(serie, order=order).fit()
            
            print("\n=============== P-VALORES DEL MODELO ===============")
            tabla = self.modelo_statsmodels.summary().tables[1].as_text()
            print(tabla)
            print("===========================================================\n")
            
        except Exception as e:
            print(f"Error fitting statsmodels model: {e}")
            self.modelo_statsmodels = None
        
        return self.modelo
    
    def obtener_pvalores(self):
        """Get p-values of model parameters"""
        if self.modelo_statsmodels is None:
            return "No statsmodels model fitted. Cannot obtain p-values."
        
        try:
            resultados = self.modelo_statsmodels.summary()
            tabla_pvalores = resultados.tables[1]
            
            parametros_significativos = 0
            total_parametros = 0
            
            tabla_texto = tabla_pvalores.as_text()
            
            lineas = tabla_texto.split('\n')
            tabla_formateada = "P-VALORES DE LOS PARÁMETROS:\n"
            
            for linea in lineas:
                if 'P>|z|' in linea or 'P>|t|' in linea or '===' in linea or 'coef' in linea:
                    tabla_formateada += linea + "\n"
                elif len(linea.strip()) > 0 and not linea.strip().startswith("=="):
                    partes = linea.split()
                    if len(partes) >= 4:
                        param_name = partes[0]
                        try:
                            p_valor = float(partes[4]) if partes[4] != 'nan' else 1.0
                            
                            total_parametros += 1
                            if p_valor < 0.05:
                                parametros_significativos += 1
                                tabla_formateada += f"{linea} *** SIGNIFICATIVO ***\n"
                            else:
                                tabla_formateada += f"{linea} (no significativo)\n"
                        except (ValueError, IndexError):
                            tabla_formateada += f"{linea}\n"
            
            if total_parametros > 0:
                tabla_formateada += f"\nResumen: {parametros_significativos} de {total_parametros} "
                tabla_formateada += f"parámetros son estadísticamente significativos (p < 0.05).\n"
            
            return tabla_formateada
            
        except Exception as e:
            return f"Error obtaining p-values: {e}"
    
    def evaluar_prediccion(self, datos_reales, predicciones):
        """Evaluate prediction quality with multiple metrics"""
        mse = mean_squared_error(datos_reales, predicciones)
        rmse = np.sqrt(mse)
        
        try:
            mae = np.mean(np.abs(datos_reales - predicciones))
        except:
            mae = np.nan
        
        try:
            r2 = r2_score(datos_reales, predicciones)
        except:
            r2 = 0.0
        
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.visualizar_datos(None, "Comparación de Datos Reales vs. Predicciones", 
                             datos_reales, predicciones)
        
        return mse, rmse, mae, r2
    
    def predecir_longitud(self, dias_a_predecir):
        """Predict length for specified days"""
        if self.serie_datos is None:
            raise ValueError("Must load data first.")
        
        serie = self.serie_datos.dropna().astype(float)
        
        if len(serie) < 10:
            raise ValueError("Time series has too few data points for modeling (minimum 10).")
        
        serie_diff, d_recomendado = self.verificar_estacionariedad(serie)
        self.encontrar_mejor_modelo(serie, d_recomendado)
        
        print(f"\n============= RESUMEN DEL MODELO SELECCIONADO =============")
        print(f"Modelo seleccionado: {self.modelo}")
        print(f"Orden ARIMA: {self.modelo.order} (p,d,q)")
        
        print("\nP-valores de los parámetros del modelo:")
        p_valores_info = self.obtener_pvalores()
        print(p_valores_info)
        print("===========================================================\n")
        
        print("\nAnalizando residuos del modelo...")
        self.analizar_residuos()
        
        predicciones = self.modelo.predict(n_periods=dias_a_predecir)
        
        indices_pred = pd.RangeIndex(start=len(serie), stop=len(serie) + dias_a_predecir)
        predicciones = pd.Series(predicciones, index=indices_pred)
        
        # Apply biological limits and ensure coherent growth
        for i in range(len(predicciones)):
            if predicciones.iloc[i] > self.LIMITE_BIOLOGICO:
                predicciones.iloc[i] = self.LIMITE_BIOLOGICO
            
            if i > 0 and predicciones.iloc[i] < predicciones.iloc[i-1]:
                predicciones.iloc[i] = predicciones.iloc[i-1] + min(0.001 * predicciones.iloc[i-1], 0.05)
            
            if predicciones.iloc[i] > self.LIMITE_BIOLOGICO * 0.95:
                if i > 0:
                    distancia_al_limite = self.LIMITE_BIOLOGICO - predicciones.iloc[i-1]
                    crecimiento = distancia_al_limite * 0.05
                    predicciones.iloc[i] = predicciones.iloc[i-1] + crecimiento
        
        return predicciones

class InterfazPredictor:
    def __init__(self, root):
        self.root = root
        self.predictor = PredictorPeces()
        self.configurar_interfaz()
    
    def configurar_interfaz(self):
        """Configure GUI elements"""
        self.root.title("Predicción Adaptable de Longitud de Peces - Modelo ARIMA")
        self.root.geometry("900x800")
        self.root.configure(bg="#f0f4f8")
        
        # Create main frame with scrollbar
        main_canvas = tk.Canvas(self.root, bg="#f0f4f8")
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas, bg="#f0f4f8")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_frame = tk.Frame(scrollable_frame, bg="#f0f4f8", padx=25, pady=25)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        titulo = tk.Label(main_frame, text="Sistema de Predicción de Crecimiento de Truchas (ARIMA)", 
                         font=("Arial", 18, "bold"), bg="#f0f4f8", fg="#1a5276")
        titulo.pack(pady=15)
        
        # Subtitle
        subtitulo = tk.Label(main_frame, 
                           text=f"(Respetando el límite biológico de {self.predictor.LIMITE_BIOLOGICO} cm)", 
                           font=("Arial", 12), bg="#f0f4f8", fg="#2874a6")
        subtitulo.pack()
        
        # Data input frame
        datos_frame = tk.Frame(main_frame, bg="#e1e8ed", padx=15, pady=15, bd=2, relief=tk.GROOVE)
        datos_frame.pack(fill=tk.X, pady=10)
        
        # Load file button
        self.btn_cargar = tk.Button(datos_frame, text="Cargar Archivo Excel", command=self.cargar_archivo, 
                               font=("Arial", 12), bg="#3498db", fg="white", padx=15, pady=5)
        self.btn_cargar.pack(pady=10)
        
        # File label
        self.lbl_archivo = tk.Label(datos_frame, text="Ningún archivo seleccionado", 
                               font=("Arial", 10), bg="#e1e8ed", fg="#566573")
        self.lbl_archivo.pack(pady=5)
        
        # Days input
        dias_frame = tk.Frame(datos_frame, bg="#e1e8ed")
        dias_frame.pack(pady=10)
        
        tk.Label(dias_frame, text="Días a predecir:", font=("Arial", 12), bg="#e1e8ed").pack(side=tk.LEFT, padx=5)
        
        self.entry_dias = tk.Entry(dias_frame, font=("Arial", 12), width=10, bd=2)
        self.entry_dias.pack(side=tk.LEFT, padx=5)
        self.entry_dias.insert(0, "30")
        
        # Analysis buttons
        btns_frame = tk.Frame(main_frame, bg="#f0f4f8", pady=10)
        btns_frame.pack(fill=tk.X)
        
        self.btn_predecir = tk.Button(btns_frame, text="Realizar Predicción", command=self.realizar_prediccion, 
                                 font=("Arial", 14, "bold"), bg="#2ecc71", fg="white", padx=20, pady=10)
        self.btn_predecir.pack(side=tk.LEFT, padx=10)
        
        self.btn_autocorr = tk.Button(btns_frame, text="Ver Autocorrelación", command=self.mostrar_autocorrelacion, 
                                 font=("Arial", 14), bg="#9b59b6", fg="white", padx=20, pady=10)
        self.btn_autocorr.pack(side=tk.LEFT, padx=10)
        
        # Statistics frame
        estadisticas_frame = tk.Frame(main_frame, bg="#e5e8e8", padx=15, pady=15, bd=2, relief=tk.GROOVE)
        estadisticas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(estadisticas_frame, text="Estadísticas y P-Valores del Modelo", font=("Arial", 14, "bold"), 
               bg="#e5e8e8", fg="#34495e").pack(pady=5)
        
        # Statistics text area
        stats_frame = tk.Frame(estadisticas_frame)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.estadisticas_text = tk.Text(stats_frame, font=("Consolas", 10), height=15, width=80)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.estadisticas_text.yview)
        self.estadisticas_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.estadisticas_text.pack(side="left", fill="both", expand=True)
        stats_scrollbar.pack(side="right", fill="y")
        
        # Results frame
        resultados_frame = tk.Frame(main_frame, bg="#eaecee", padx=15, pady=15, bd=2, relief=tk.GROOVE)
        resultados_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(resultados_frame, text="Resultados de Predicción", font=("Arial", 14, "bold"), 
               bg="#eaecee", fg="#34495e").pack(pady=5)
        
        # Results text area
        results_frame = tk.Frame(resultados_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.resultado_text = tk.Text(results_frame, font=("Consolas", 11), height=12, width=80)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.resultado_text.yview)
        self.resultado_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.resultado_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Listo", bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                              font=("Arial", 10), bg="#d6eaf8", fg="#2c3e50")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def cargar_archivo(self):
        """Handle file loading"""
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
        """Show autocorrelation plots for loaded data"""
        if not hasattr(self, 'archivo_path'):
            messagebox.showerror("Error", "Debe seleccionar un archivo primero.")
            return
        
        try:
            if self.predictor.serie_datos is None:
                if not self.predictor.cargar_datos(self.archivo_path):
                    messagebox.showerror("Error", "No se pudieron cargar los datos correctamente.")
                    return
            
            serie = self.predictor.serie_datos.dropna().astype(float)
            
            messagebox.showinfo("Análisis de Autocorrelación", 
                             "Se mostrarán gráficos de ACF y PACF para ayudar a determinar los órdenes p y q del modelo ARIMA.")
            
            self.predictor.verificar_estacionariedad(serie)
            
            if self.predictor.modelo_statsmodels is not None:
                self.predictor.analizar_residuos()
            
            self.status_bar.config(text="Análisis de autocorrelación completado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el análisis: {str(e)}")
            self.status_bar.config(text="Error en el análisis de autocorrelación")
    
    def realizar_prediccion(self):
        """Perform prediction process"""
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
            
            if not self.predictor.cargar_datos(self.archivo_path):
                messagebox.showerror("Error", "No se pudieron cargar los datos correctamente.")
                return
            
            self.status_bar.config(text="Realizando predicción...")
            self.root.update()
            
            serie = self.predictor.serie_datos
            train_size = int(len(serie) * 0.8)
            train, test = serie[:train_size], serie[train_size:]
            
            # Capture console output
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            predicciones = self.predictor.predecir_longitud(dias)
            
            sys.stdout = old_stdout
            output = new_stdout.getvalue()
            
            self.mostrar_estadisticas(output)
            
            test_compare = test[:min(dias, len(test))]
            if len(test_compare) > 0:
                mse, rmse, mae, r2 = self.predictor.evaluar_prediccion(test_compare, predicciones[:len(test_compare)])
                self.mostrar_resultados(predicciones, mse, rmse, mae, r2)
            else:
                self.mostrar_resultados(predicciones)
            
            messagebox.showinfo("Análisis Completado", 
                             "La predicción se ha completado con éxito.\n\n"
                             "Se han generado gráficos de autocorrelación durante el análisis.")
            
            self.status_bar.config(text="Predicción completada respetando límites biológicos.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante la predicción: {str(e)}")
            self.status_bar.config(text="Error en la predicción")
    
    def mostrar_estadisticas(self, output):
        """Show statistical information and p-values"""
        self.estadisticas_text.delete(1.0, tk.END)
        
        self.estadisticas_text.insert(tk.END, "ANÁLISIS ESTADÍSTICO DEL MODELO ARIMA\n")
        self.estadisticas_text.insert(tk.END, "==========================================\n\n")
        
        lines = output.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if "ANÁLISIS DE ESTACIONARIEDAD" in line:
                self.estadisticas_text.insert(tk.END, "ANÁLISIS DE ESTACIONARIEDAD:\n")
                i += 1
                while i < len(lines) and "=====" not in lines[i]:
                    if "P-VALOR" in lines[i]:
                        self.estadisticas_text.insert(tk.END, lines[i] + "\n")
                    i += 1
                self.estadisticas_text.insert(tk.END, "\n")
            
            elif "P-VALORES DEL MODELO" in line:
                self.estadisticas_text.insert(tk.END, "P-VALORES DEL MODELO ARIMA:\n")
                self.estadisticas_text.insert(tk.END, "-------------------------------------------\n")
                i += 1
                while i < len(lines) and "=====" not in lines[i]:
                    self.estadisticas_text.insert(tk.END, lines[i] + "\n")
                    i += 1
                self.estadisticas_text.insert(tk.END, "\n")
            
            elif "RESUMEN DEL MODELO SELECCIONADO" in line:
                self.estadisticas_text.insert(tk.END, "INFORMACIÓN DEL MODELO:\n")
                i += 1
                while i < len(lines) and "P-valores" not in lines[i]:
                    if "ARIMA" in lines[i] or "Modelo" in lines[i]:
                        self.estadisticas_text.insert(tk.END, lines[i] + "\n")
                    i += 1
                self.estadisticas_text.insert(tk.END, "\n")
            
            i += 1
        
        self.estadisticas_text.insert(tk.END, "\nACERCA DE LOS GRÁFICOS DE AUTOCORRELACIÓN:\n")
        self.estadisticas_text.insert(tk.END, "- ACF: Muestra relaciones entre observaciones separadas por k periodos\n")
        self.estadisticas_text.insert(tk.END, "- PACF: Muestra correlaciones parciales eliminando efectos intermedios\n")
        self.estadisticas_text.insert(tk.END, "- ACF significativa en rezago q → Orden MA(q)\n")
        self.estadisticas_text.insert(tk.END, "- PACF significativa en rezago p → Orden AR(p)\n\n")
        
        self.estadisticas_text.insert(tk.END, "\nINTERPRETACIÓN DE P-VALORES:\n")
        self.estadisticas_text.insert(tk.END, "- p < 0.05: El parámetro es estadísticamente significativo\n")
        self.estadisticas_text.insert(tk.END, "- p ≥ 0.05: El parámetro podría no ser necesario en el modelo\n\n")
    
    def mostrar_resultados(self, predicciones, mse=None, rmse=None, mae=None, r2=None):
        """Show results in text area"""
        self.resultado_text.delete(1.0, tk.END)
    
        if mse is not None:
            self.resultado_text.insert(tk.END, f"Métricas de evaluación:\n")
            self.resultado_text.insert(tk.END, f"Error Cuadrático Medio (MSE): {mse:.4f}\n")
            self.resultado_text.insert(tk.END, f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}\n")
            
            if np.isnan(mae):
                self.resultado_text.insert(tk.END, f"Error Absoluto Medio (MAE): No disponible\n")
            else:
                self.resultado_text.insert(tk.END, f"Error Absoluto Medio (MAE): {mae:.4f}\n")
            
            self.resultado_text.insert(tk.END, f"R Cuadrado (R²): {r2:.4f}\n\n")
        
        self.resultado_text.insert(tk.END, "Predicciones para los próximos días:\n")
        self.resultado_text.insert(tk.END, f"(Límite biológico: {self.predictor.LIMITE_BIOLOGICO} cm)\n\n")
        
        for i, valor in enumerate(predicciones.values):
            if valor >= self.predictor.LIMITE_BIOLOGICO * 0.95:
                self.resultado_text.insert(tk.END, f"Día {i+1}: {valor:.2f} cm (¡Cerca del límite biológico!)\n")
            else:
                self.resultado_text.insert(tk.END, f"Día {i+1}: {valor:.2f} cm\n")
            
            if i > 0:
                crecimiento = valor - predicciones.values[i-1]
                self.resultado_text.insert(tk.END, f"   Crecimiento: +{crecimiento:.4f} cm\n")

# Main entry point
if __name__ == "__main__":
    print("Iniciando Sistema de Predicción ARIMA con Análisis de P-valores y Autocorrelación...")
    
    try:
        root = tk.Tk()
        app = InterfazPredictor(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        input("Press Enter to exit...")
