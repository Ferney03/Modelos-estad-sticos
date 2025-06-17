import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import statsmodels.api as sm

class FishLengthPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelo de Predicción de Longitud de Peces")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        self.data = None
        self.model = None
        self.max_day = 0
        self.feature_names = []
        self.p_values = []  # Añadimos un atributo para almacenar los valores p
        
        # Crear elementos de UI
        self.create_widgets()
    
    def create_widgets(self):
        # Marco para selección de archivo
        file_frame = tk.Frame(self.root, bg="#f0f0f0")
        file_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(file_frame, text="Seleccionar Archivo Excel:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        self.file_path_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side="left", padx=5)
        tk.Button(file_frame, text="Explorar", command=self.browse_file).pack(side="left", padx=5)
        
        # Marco para configuración de predicción
        config_frame = tk.Frame(self.root, bg="#f0f0f0")
        config_frame.pack(fill="x", padx=20, pady=10)
        
        # Días a predecir
        days_frame = tk.Frame(config_frame, bg="#f0f0f0")
        days_frame.pack(side="left", padx=20)
        tk.Label(days_frame, text="Días a predecir:", bg="#f0f0f0").pack(anchor="w")
        self.days_var = tk.StringVar(value="30")
        tk.Entry(days_frame, textvariable=self.days_var, width=10).pack(anchor="w", pady=5)
        
        # Opciones de restricción de tasa de crecimiento
        constraint_frame = tk.Frame(config_frame, bg="#f0f0f0")
        constraint_frame.pack(side="left", padx=20)
        tk.Label(constraint_frame, text="Restricción de tasa de crecimiento:", bg="#f0f0f0").pack(anchor="w")
        self.constraint_var = tk.StringVar(value="historical")
        ttk.Radiobutton(constraint_frame, text="Usar tasa histórica de crecimiento", variable=self.constraint_var, 
                       value="historical").pack(anchor="w")
        ttk.Radiobutton(constraint_frame, text="Usar predicción del modelo con suavizado", variable=self.constraint_var, 
                       value="smoothed").pack(anchor="w")
        
        # Periodo de análisis para tasa histórica de crecimiento
        period_frame = tk.Frame(config_frame, bg="#f0f0f0")
        period_frame.pack(side="left", padx=20)
        tk.Label(period_frame, text="Periodo de análisis (días):", bg="#f0f0f0").pack(anchor="w")
        self.period_var = tk.StringVar(value="30")
        tk.Entry(period_frame, textvariable=self.period_var, width=10).pack(anchor="w", pady=5)
        
        # Botón para ejecutar predicción
        tk.Button(self.root, text="Ejecutar Predicción", command=self.run_prediction, 
                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), 
                 padx=10, pady=5).pack(pady=10)
        
        # Notebook para resultados
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Pestaña para métricas y coeficientes
        self.metrics_tab = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.metrics_tab, text="Métricas del Modelo")
        
        # Widget de texto para mostrar métricas
        self.metrics_text = tk.Text(self.metrics_tab, height=10, width=80)
        self.metrics_text.pack(pady=10, fill="both", expand=True)
        
        # Pestaña para tabla de predicciones
        self.predictions_tab = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.predictions_tab, text="Tabla de Predicciones")
        
        # Tabla para predicciones
        self.predictions_frame = tk.Frame(self.predictions_tab, bg="#f0f0f0")
        self.predictions_frame.pack(fill="both", expand=True, pady=10)
        
        # Pestaña para el gráfico
        self.plot_tab = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.plot_tab, text="Gráfico de Crecimiento")
        
        # Marco para el gráfico
        self.plot_frame = tk.Frame(self.plot_tab, bg="#f0f0f0")
        self.plot_frame.pack(fill="both", expand=True, pady=10)
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos Excel", "*.xlsx;*.xls")])
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"Archivo seleccionado: {os.path.basename(file_path)}")
    
    def load_data(self):
        try:
            file_path = self.file_path_var.get()
            if not file_path:
                messagebox.showerror("Error", "Por favor seleccione un archivo Excel")
                return False
            
            self.status_var.set("Cargando datos...")
            self.root.update()
            
            self.data = pd.read_excel(file_path)
            
            # Verificar si existen las columnas requeridas
            required_columns = ["Tiempo_dias", "Longitud_cm"]
            for col in required_columns:
                if col not in self.data.columns:
                    messagebox.showerror("Error", f"No se encontró la columna requerida '{col}' en el archivo Excel")
                    return False
            
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
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            return False
    
    def train_model(self):
        try:
            self.status_var.set("Entrenando modelo...")
            self.root.update()
            
            # Preparar características (X) y objetivo (y)
            X = self.data[self.feature_names]
            y = self.data["Longitud_cm"]
            
            # Crear y entrenar el modelo de scikit-learn
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            # Calcular métricas en datos de entrenamiento
            y_pred = self.model.predict(X)
            self.mse = mean_squared_error(y, y_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y, y_pred)
            self.r2 = r2_score(y, y_pred)
            
            # Almacenar predicciones para datos de entrenamiento
            self.data["Predicted_Length"] = y_pred
            
            # Usar statsmodels para obtener valores p
            # Agregar constante para intercepto
            X_sm = sm.add_constant(X)
            # Ajustar modelo de statsmodels
            model_sm = sm.OLS(y, X_sm).fit()
            
            # Extraer valores p
            self.p_values = model_sm.pvalues
            # Guardar el resumen del modelo para información adicional
            self.model_summary = model_sm.summary()
            
            self.status_var.set("Modelo entrenado con éxito")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")
            self.status_var.set("Error al entrenar el modelo")
            return False
    
    def get_historical_growth_rate(self, analysis_period):
        try:
            # Obtener los puntos de datos más recientes basados en el periodo de análisis
            recent_data = self.data.tail(analysis_period)
            
            # Calcular la tasa de crecimiento diario promedio
            total_growth = recent_data["Longitud_cm"].iloc[-1] - recent_data["Longitud_cm"].iloc[0]
            total_days = recent_data["Tiempo_dias"].iloc[-1] - recent_data["Tiempo_dias"].iloc[0]
            
            if total_days == 0:
                return 0
                
            avg_daily_growth = total_growth / total_days
            
            # Calcular la desviación estándar del crecimiento diario
            daily_growth_rates = recent_data["Growth_Rate"].dropna()
            std_daily_growth = daily_growth_rates.std()
            
            return {
                "avg_rate": avg_daily_growth,
                "std_rate": std_daily_growth,
                "max_rate": daily_growth_rates.max(),
                "min_rate": daily_growth_rates.min(),
                "recent_rate": daily_growth_rates.iloc[-5:].mean() if len(daily_growth_rates) >= 5 else avg_daily_growth
            }
        except Exception as e:
            print(f"Error al calcular la tasa histórica de crecimiento: {str(e)}")
            return {"avg_rate": 0.01, "std_rate": 0.005, "max_rate": 0.02, "min_rate": 0, "recent_rate": 0.01}
    
    def predict_future(self, days_to_predict):
        try:
            self.status_var.set("Prediciendo valores futuros...")
            self.root.update()
            
            # Obtener la última fila de datos como punto de partida
            last_row = self.data.iloc[-1:].copy()
            last_length = last_row["Longitud_cm"].values[0]
            
            # Obtener periodo de análisis
            try:
                analysis_period = int(self.period_var.get())
                if analysis_period <= 0:
                    analysis_period = 30
            except ValueError:
                analysis_period = 30
            
            # Obtener estadísticas de tasa histórica de crecimiento
            growth_stats = self.get_historical_growth_rate(analysis_period)
            
            # Crear un dataframe para almacenar predicciones
            future_data = pd.DataFrame(columns=self.data.columns)
            
            # Añadir el último punto de datos conocido
            future_data = pd.concat([future_data, last_row])
            
            # Generar predicciones para días futuros
            constraint_type = self.constraint_var.get()
            
            for i in range(1, days_to_predict + 1):
                new_day = self.max_day + i
                new_row = last_row.copy()
                new_row["Tiempo_dias"] = new_day
                
                # Predecir longitud usando el modelo
                features = new_row[self.feature_names]
                model_prediction = self.model.predict(features)[0]
                
                if constraint_type == "historical":
                    # Usar tasa histórica de crecimiento con pequeña variación aleatoria
                    # Enfatizar la tendencia de crecimiento más reciente
                    random_factor = np.random.normal(0, growth_stats["std_rate"] / 3)
                    growth_rate = growth_stats["recent_rate"] + random_factor
                    
                    # Asegurar que la tasa de crecimiento esté dentro de límites históricos
                    growth_rate = max(growth_stats["min_rate"], min(growth_stats["max_rate"], growth_rate))
                    
                    # Calcular nueva longitud basada en tasa histórica de crecimiento
                    new_length = last_length + growth_rate
                    
                    # Mezclar con predicción del modelo (90% histórico, 10% modelo)
                    predicted_length = 0.9 * new_length + 0.1 * model_prediction
                    
                elif constraint_type == "smoothed":
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
                
                # Aplicar restricción realista para trucha arcoíris (máx 70cm)
                predicted_length = min(predicted_length, 70.0)
                
                # Asegurar que el pez no encoja
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
            
            self.status_var.set("Predicción completada")
            return future_data.iloc[1:]  # Omitir la primera fila (que era el último punto de datos conocido)
        except Exception as e:
            messagebox.showerror("Error", f"Error al predecir valores futuros: {str(e)}")
            self.status_var.set("Error en la predicción")
            return None
    
    def display_results(self, future_data):
        try:
            self.status_var.set("Mostrando resultados...")
            self.root.update()
            
            # Limpiar resultados previos
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            for widget in self.predictions_frame.winfo_children():
                widget.destroy()
            
            self.metrics_text.delete(1.0, tk.END)
            
            # Mostrar métricas
            metrics_str = f"Métricas de Rendimiento del Modelo:\n"
            metrics_str += f"R² (Coeficiente de Determinación): {self.r2:.6f} ({self.r2 * 100:.2f}%)\n"
            metrics_str += f"Error Cuadrático Medio (MSE): {self.mse:.6f}\n"
            metrics_str += f"Raíz del Error Cuadrático Medio (RMSE): {self.rmse:.6f}\n"
            metrics_str += f"Error Absoluto Medio (MAE): {self.mae:.6f}\n\n"
            
            # Mostrar estadísticas de tasa de crecimiento
            try:
                analysis_period = int(self.period_var.get())
                if analysis_period <= 0:
                    analysis_period = 30
            except ValueError:
                analysis_period = 30
                
            growth_stats = self.get_historical_growth_rate(analysis_period)
            
            metrics_str += f"Estadísticas de Tasa de Crecimiento (últimos {analysis_period} días):\n"
            metrics_str += f"Crecimiento diario promedio: {growth_stats['avg_rate']:.6f} cm/día\n"
            metrics_str += f"Crecimiento diario reciente: {growth_stats['recent_rate']:.6f} cm/día\n"
            metrics_str += f"Desviación estándar: {growth_stats['std_rate']:.6f} cm/día\n"
            metrics_str += f"Crecimiento diario mínimo: {growth_stats['min_rate']:.6f} cm/día\n"
            metrics_str += f"Crecimiento diario máximo: {growth_stats['max_rate']:.6f} cm/día\n\n"
            
            # Mostrar coeficientes y valores p
            metrics_str += "Coeficientes del Modelo y Valores P:\n"
            metrics_str += f"Intercepto: {self.model.intercept_:.6f} (p-valor: {self.p_values['const']:.6f})\n"
            
            # Mostrar coeficientes y valores p para cada característica
            for i, feature in enumerate(self.feature_names):
                coef = self.model.coef_[i]
                p_value = self.p_values[feature]
                significance = ""
                if p_value < 0.001:
                    significance = "***"  # Altamente significativo
                elif p_value < 0.01:
                    significance = "**"   # Muy significativo
                elif p_value < 0.05:
                    significance = "*"    # Significativo
                elif p_value < 0.1:
                    significance = "."    # Marginalmente significativo
                
                metrics_str += f"{feature}: {coef:.6f} (p-valor: {p_value:.6f}) {significance}\n"
            
            # Añadir leyenda para los símbolos de significancia
            metrics_str += "\nLeyenda de significancia estadística:\n"
            metrics_str += "*** p<0.001 (altamente significativo)\n"
            metrics_str += "** p<0.01 (muy significativo)\n"
            metrics_str += "* p<0.05 (significativo)\n"
            metrics_str += ". p<0.1 (marginalmente significativo)\n"
            
            self.metrics_text.insert(tk.END, metrics_str)
            
            # Crear tabla de predicciones
            columns = ["Día", "Longitud Predicha (cm)", "Crecimiento Diario (cm)"]
            tree = ttk.Treeview(self.predictions_frame, columns=columns, show="headings")
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Añadir datos a la tabla
            for i, row in future_data.iterrows():
                day = int(row["Tiempo_dias"])
                length = row["Longitud_cm"]
                growth = row["Growth_Rate"]
                tree.insert("", "end", values=(day, f"{length:.4f}", f"{growth:.6f}"))
            
            # Añadir barra de desplazamiento
            scrollbar = ttk.Scrollbar(self.predictions_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            tree.pack(expand=True, fill="both")
            
            # Crear una figura para graficar
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            
            # Graficar datos originales
            ax.scatter(self.data["Tiempo_dias"], self.data["Longitud_cm"], 
                      color='blue', label='Datos Originales', s=30, alpha=0.7)
            
            # Graficar datos futuros predichos
            ax.scatter(future_data["Tiempo_dias"], future_data["Longitud_cm"], 
                      color='red', label='Datos Predichos', s=30, alpha=0.7)
            
            # Conectar los puntos con líneas
            all_days = list(self.data["Tiempo_dias"]) + list(future_data["Tiempo_dias"])
            all_lengths = list(self.data["Longitud_cm"]) + list(future_data["Longitud_cm"])
            ax.plot(all_days, all_lengths, 'k--', alpha=0.5)
            
            # Añadir un recuadro ampliado para el área de predicción
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
            
            # Solo crear recuadro ampliado si tenemos suficientes datos
            if len(self.data) > 10:
                # Crear ejes de recuadro ampliado
                axins = zoomed_inset_axes(ax, zoom=2.5, loc='lower right')
                
                # Determinar región de zoom (últimos 10 puntos originales + predicciones)
                zoom_start = max(0, len(self.data) - 10)
                zoom_days = list(self.data["Tiempo_dias"].iloc[zoom_start:]) + list(future_data["Tiempo_dias"])
                zoom_lengths = list(self.data["Longitud_cm"].iloc[zoom_start:]) + list(future_data["Longitud_cm"])
                
                # Graficar datos en recuadro ampliado
                axins.scatter(self.data["Tiempo_dias"].iloc[zoom_start:], self.data["Longitud_cm"].iloc[zoom_start:], 
                             color='blue', s=20, alpha=0.7)
                axins.scatter(future_data["Tiempo_dias"], future_data["Longitud_cm"], 
                             color='red', s=20, alpha=0.7)
                axins.plot(zoom_days, zoom_lengths, 'k--', alpha=0.5)
                
                # Establecer límites para recuadro ampliado
                x_min = min(zoom_days) - 1
                x_max = max(zoom_days) + 1
                y_min = min(zoom_lengths) - 0.5
                y_max = max(zoom_lengths) + 0.5
                axins.set_xlim(x_min, x_max)
                axins.set_ylim(y_min, y_max)
                
                # Desactivar marcas de recuadro ampliado
                axins.tick_params(labelleft=False, labelbottom=False)
                
                # Dibujar líneas conectoras entre recuadro ampliado y gráfico principal
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
            ax.set_xlabel('Tiempo (días)')
            ax.set_ylabel('Longitud (cm)')
            ax.set_title('Predicción de Longitud de Peces')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Incrustar el gráfico en la ventana tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Cambiar a la pestaña del gráfico
            self.notebook.select(self.plot_tab)
            
            self.status_var.set("Resultados mostrados con éxito")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar resultados: {str(e)}")
            self.status_var.set("Error al mostrar resultados")
    
    def run_prediction(self):
        # Cargar datos
        if not self.load_data():
            return
        
        # Entrenar modelo
        if not self.train_model():
            return
        
        # Obtener número de días a predecir
        try:
            days_to_predict = int(self.days_var.get())
            if days_to_predict <= 0:
                messagebox.showerror("Error", "Los días a predecir deben ser un entero positivo")
                return
        except ValueError:
            messagebox.showerror("Error", "Los días a predecir deben ser un entero válido")
            return
        
        # Predecir valores futuros
        future_data = self.predict_future(days_to_predict)
        if future_data is None:
            return
        
        # Mostrar resultados
        self.display_results(future_data)
        
        # Mostrar resumen de predicción
        last_day = self.max_day + days_to_predict
        last_length = future_data.iloc[-1]["Longitud_cm"]
        messagebox.showinfo("Predicción Completada", 
                           f"¡Predicción completada con éxito!\n\n"
                           f"En el día {last_day}, la longitud predicha del pez es {last_length:.4f} cm.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FishLengthPredictor(root)
    root.mainloop()