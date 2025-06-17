import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import matplotlib
matplotlib.use('TkAgg')
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import norm

class VARFishPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Modelo VAR")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.LIMITE_BIOLOGICO = 70  # Límite biológico en cm para truchas
        
        # Crear elementos de UI
        self.create_widgets()
    
    def create_widgets(self):
        # Estilo para ttk
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 10), background='#3498db')
        style.configure('TNotebook', background='#f5f8fa')
        style.configure('TNotebook.Tab', padding=[10, 5], font=('Arial', 10))
        style.configure('Treeview', font=('Arial', 9))
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        
        # Marco principal
        main_frame = tk.Frame(self.root, bg="#f5f8fa", padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)
        
        # Título
        title_frame = tk.Frame(main_frame, bg="#f5f8fa")
        title_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(title_frame, text="Sistema Avanzado de Predicción de Crecimiento de Peces", 
                font=("Arial", 16, "bold"), bg="#f5f8fa", fg="#2c3e50").pack()
        
        tk.Label(title_frame, text="Modelo VAR (Vector Autoregressive)", 
                font=("Arial", 12), bg="#f5f8fa", fg="#34495e").pack(pady=(5, 0))
        
        # Marco para selección de archivo y configuración
        config_frame = tk.LabelFrame(main_frame, text="Configuración", font=("Arial", 11, "bold"), 
                                   bg="#f5f8fa", fg="#2c3e50", padx=15, pady=15)
        config_frame.pack(fill="x", pady=(0, 15))
        
        # Selección de archivo
        file_frame = tk.Frame(config_frame, bg="#f5f8fa")
        file_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(file_frame, text="Archivo Excel:", bg="#f5f8fa", font=("Arial", 10)).pack(side="left", padx=(0, 5))
        self.file_path_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.file_path_var, width=50).pack(side="left", padx=(0, 5))
        ttk.Button(file_frame, text="Explorar", command=self.browse_file).pack(side="left")
        
        # Marco para opciones
        options_frame = tk.Frame(config_frame, bg="#f5f8fa")
        options_frame.pack(fill="x")
        
        # Columna 1: Configuración básica
        col1 = tk.Frame(options_frame, bg="#f5f8fa")
        col1.pack(side="left", padx=(0, 20))
        
        # Variable objetivo
        tk.Label(col1, text="Variable objetivo:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        self.target_var = tk.StringVar(value="Longitud_cm")
        tk.Entry(col1, textvariable=self.target_var, width=15).pack(anchor="w")
        
        # Días a predecir
        tk.Label(col1, text="Días a predecir:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(10, 5))
        self.days_var = tk.StringVar(value="30")
        tk.Entry(col1, textvariable=self.days_var, width=10).pack(anchor="w")
        
        # Columna 2: Parámetros del modelo VAR
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Parámetros del modelo VAR:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Orden máximo del modelo VAR
        maxlag_frame = tk.Frame(col2, bg="#f5f8fa")
        maxlag_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(maxlag_frame, text="Orden máximo (maxlag):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.maxlag_var = tk.StringVar(value="15")
        tk.Entry(maxlag_frame, textvariable=self.maxlag_var, width=8).pack(side="left")
        
        # Criterio de selección
        criteria_frame = tk.Frame(col2, bg="#f5f8fa")
        criteria_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(criteria_frame, text="Criterio de selección:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.criteria_var = tk.StringVar(value="aic")
        criteria_combo = ttk.Combobox(criteria_frame, textvariable=self.criteria_var, width=8)
        criteria_combo['values'] = ('aic', 'bic', 'hqic', 'fpe')
        criteria_combo.pack(side="left")
        
        # Diferenciación
        diff_frame = tk.Frame(col2, bg="#f5f8fa")
        diff_frame.pack(anchor="w")
        
        tk.Label(diff_frame, text="Diferenciación:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.diff_var = tk.StringVar(value="auto")
        diff_combo = ttk.Combobox(diff_frame, textvariable=self.diff_var, width=8)
        diff_combo['values'] = ('auto', '0', '1', '2')
        diff_combo.pack(side="left")
        
        # Columna 3: Opciones avanzadas
        col3 = tk.Frame(options_frame, bg="#f5f8fa")
        col3.pack(side="left")
        
        tk.Label(col3, text="Opciones avanzadas:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Normalización de datos
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Normalizar datos", variable=self.normalize_var).pack(anchor="w")
        
        # Selección automática de variables
        self.auto_select_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Selección automática de variables", variable=self.auto_select_var).pack(anchor="w")
        
        # Aplicar restricciones biológicas
        self.bio_constraints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Aplicar restricciones biológicas", variable=self.bio_constraints_var).pack(anchor="w")
        
        # Botón para ejecutar predicción
        button_frame = tk.Frame(main_frame, bg="#f5f8fa")
        button_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Button(button_frame, text="Ejecutar Predicción", command=self.run_prediction,
                  style='TButton', padding=(20, 10)).pack(pady=(10, 0))
        
        # Notebook para resultados
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Pestaña para resumen y métricas
        self.summary_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.summary_tab, text="Resumen y Métricas")
        
        # Pestaña para tabla de predicciones
        self.predictions_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.predictions_tab, text="Tabla de Predicciones")
        
        # Pestaña para el gráfico de crecimiento
        self.growth_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.growth_tab, text="Gráfico de Crecimiento")
        
        # Pestaña para análisis de series temporales
        self.timeseries_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.timeseries_tab, text="Análisis de Series Temporales")
        
        # Pestaña para diagnóstico del modelo
        self.diagnostics_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.diagnostics_tab, text="Diagnóstico del Modelo")
        
        # Configurar contenido de las pestañas
        self.setup_summary_tab()
        self.setup_predictions_tab()
        self.setup_growth_tab()
        self.setup_timeseries_tab()
        self.setup_diagnostics_tab()
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Listo para comenzar. Seleccione un archivo Excel y configure los parámetros.")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                               anchor=tk.W, font=("Arial", 9), bg="#e1e8ed", fg="#34495e", padx=10, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_summary_tab(self):
        """Configura la pestaña de resumen y métricas"""
        # Marco para resumen
        summary_frame = tk.Frame(self.summary_tab, bg="#f5f8fa", padx=15, pady=15)
        summary_frame.pack(fill="both", expand=True)
        
        # Dividir en dos columnas
        left_frame = tk.Frame(summary_frame, bg="#f5f8fa")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        right_frame = tk.Frame(summary_frame, bg="#f5f8fa")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Métricas del modelo
        metrics_frame = tk.LabelFrame(left_frame, text="Métricas del Modelo", font=("Arial", 11, "bold"), 
                                    bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        metrics_frame.pack(fill="both", expand=True)
        
        self.metrics_text = tk.Text(metrics_frame, height=15, width=40, font=("Consolas", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        self.metrics_text.pack(fill="both", expand=True, pady=(5, 0))
        
        metrics_scroll = tk.Scrollbar(metrics_frame, command=self.metrics_text.yview)
        metrics_scroll.pack(side="right", fill="y")
        self.metrics_text.config(yscrollcommand=metrics_scroll.set)
        
        # Parámetros del modelo
        params_frame = tk.LabelFrame(right_frame, text="Parámetros del Modelo", font=("Arial", 11, "bold"), 
                                   bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        params_frame.pack(fill="both", expand=True)
        
        self.params_text = tk.Text(params_frame, height=15, width=40, font=("Consolas", 10), 
                                 bg="#ffffff", fg="#333333", wrap=tk.WORD)
        self.params_text.pack(fill="both", expand=True, pady=(5, 0))
        
        params_scroll = tk.Scrollbar(params_frame, command=self.params_text.yview)
        params_scroll.pack(side="right", fill="y")
        self.params_text.config(yscrollcommand=params_scroll.set)
        
        # Explicación del modelo
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo VAR", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """El Modelo VAR (Vector Autoregressive) es una técnica estadística avanzada para modelar series temporales multivariadas. Es especialmente útil cuando:

1. Existen múltiples variables que evolucionan juntas a lo largo del tiempo
2. Las variables tienen interdependencias dinámicas entre sí
3. Se necesita capturar la estructura temporal de los datos

El modelo VAR funciona modelando cada variable como una función lineal de los valores pasados de sí misma y de las demás variables del sistema. Características principales:

- Captura las relaciones dinámicas entre múltiples variables a lo largo del tiempo
- Permite analizar cómo los cambios en una variable afectan a las demás
- Proporciona predicciones que tienen en cuenta la evolución temporal conjunta de todas las variables
- Permite realizar análisis de impulso-respuesta para estudiar el impacto de perturbaciones en el sistema

A diferencia de otros modelos, VAR considera explícitamente la estructura temporal de los datos y las interdependencias entre variables, lo que lo hace ideal para predecir el crecimiento de peces considerando múltiples factores ambientales y biológicos que cambian con el tiempo."""
        
        explanation_text.insert(tk.END, explanation)
        explanation_text.config(state="disabled")
    
    def setup_predictions_tab(self):
        """Configura la pestaña de tabla de predicciones"""
        predictions_frame = tk.Frame(self.predictions_tab, bg="#f5f8fa", padx=15, pady=15)
        predictions_frame.pack(fill="both", expand=True)
        
        # Marco para la tabla
        table_frame = tk.LabelFrame(predictions_frame, text="Predicciones Detalladas", font=("Arial", 11, "bold"), 
                                  bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        table_frame.pack(fill="both", expand=True)
        
        # Crear tabla con Treeview
        columns = ("Día", "Longitud (cm)", "Crecimiento (cm)", "Tasa (%)", "Estado")
        self.predictions_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        # Configurar encabezados y anchos de columna
        self.predictions_tree.heading("Día", text="Día")
        self.predictions_tree.column("Día", width=80, anchor="center")
        
        self.predictions_tree.heading("Longitud (cm)", text="Longitud (cm)")
        self.predictions_tree.column("Longitud (cm)", width=120, anchor="center")
        
        self.predictions_tree.heading("Crecimiento (cm)", text="Crecimiento (cm)")
        self.predictions_tree.column("Crecimiento (cm)", width=120, anchor="center")
        
        self.predictions_tree.heading("Tasa (%)", text="Tasa (%)")
        self.predictions_tree.column("Tasa (%)", width=100, anchor="center")
        
        self.predictions_tree.heading("Estado", text="Estado")
        self.predictions_tree.column("Estado", width=150, anchor="center")
        
        # Añadir barra de desplazamiento
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.predictions_tree.yview)
        self.predictions_tree.configure(yscrollcommand=table_scroll.set)
        
        # Colocar tabla y scrollbar
        table_scroll.pack(side="right", fill="y")
        self.predictions_tree.pack(side="left", fill="both", expand=True)
        
        # Marco para estadísticas de crecimiento
        stats_frame = tk.LabelFrame(predictions_frame, text="Estadísticas de Crecimiento", font=("Arial", 11, "bold"), 
                                  bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        stats_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        self.growth_stats_text = tk.Text(stats_frame, height=6, width=80, font=("Consolas", 10), 
                                       bg="#ffffff", fg="#333333")
        self.growth_stats_text.pack(fill="both", expand=True, pady=(5, 0))
    
    def setup_growth_tab(self):
        """Configura la pestaña de gráfico de crecimiento"""
        growth_frame = tk.Frame(self.growth_tab, bg="#f5f8fa", padx=15, pady=15)
        growth_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico principal
        self.growth_plot_frame = tk.LabelFrame(growth_frame, text="Gráfico de Crecimiento", font=("Arial", 11, "bold"), 
                                            bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        self.growth_plot_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de tasa de crecimiento
        self.growth_rate_frame = tk.LabelFrame(growth_frame, text="Tasa de Crecimiento Diario", font=("Arial", 11, "bold"), 
                                            bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        self.growth_rate_frame.pack(fill="both", expand=True, pady=(15, 0))
    
    def setup_timeseries_tab(self):
        """Configura la pestaña de análisis de series temporales"""
        timeseries_frame = tk.Frame(self.timeseries_tab, bg="#f5f8fa", padx=15, pady=15)
        timeseries_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de series temporales
        self.timeseries_plot_frame = tk.LabelFrame(timeseries_frame, text="Series Temporales", 
                                                font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                                padx=10, pady=10)
        self.timeseries_plot_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de autocorrelación
        self.autocorr_frame = tk.LabelFrame(timeseries_frame, text="Análisis de Autocorrelación", 
                                         font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                         padx=10, pady=10)
        self.autocorr_frame.pack(fill="both", expand=True, pady=(15, 0))
    
    def setup_diagnostics_tab(self):
        """Configura la pestaña de diagnóstico del modelo"""
        diagnostics_frame = tk.Frame(self.diagnostics_tab, bg="#f5f8fa", padx=15, pady=15)
        diagnostics_frame.pack(fill="both", expand=True)
        
        # Marco para los gráficos de diagnóstico
        self.diagnostics_plot_frame = tk.LabelFrame(diagnostics_frame, text="Diagnóstico del Modelo", 
                                                font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                                padx=10, pady=10)
        self.diagnostics_plot_frame.pack(fill="both", expand=True)
        
        # Marco para la tabla de criterios de información
        self.info_criteria_frame = tk.LabelFrame(diagnostics_frame, text="Criterios de Información", 
                                              font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                              padx=10, pady=10)
        self.info_criteria_frame.pack(fill="both", expand=True, pady=(15, 0))
    
    def browse_file(self):
        """Abre un diálogo para seleccionar un archivo Excel"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Archivo Excel",
            filetypes=[("Archivos Excel", "*.xlsx;*.xls"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"Archivo seleccionado: {os.path.basename(file_path)}")
    
    def load_data(self):
        """Carga y preprocesa los datos desde el archivo Excel seleccionado"""
        try:
            file_path = self.file_path_var.get()
            if not file_path:
                messagebox.showerror("Error", "Por favor seleccione un archivo Excel")
                return False
            
            self.status_var.set("Cargando datos...")
            self.root.update()
            
            # Cargar datos
            self.data = pd.read_excel(file_path)
            
            # Verificar si existe la columna objetivo
            target_column = self.target_var.get()
            if target_column not in self.data.columns:
                messagebox.showerror("Error", f"No se encontró la columna objetivo '{target_column}' en el archivo Excel")
                return False
            
            # Reemplazar comas con puntos en columnas numéricas y convertir a float
            for col in self.data.columns:
                if self.data[col].dtype == object:
                    try:
                        self.data[col] = self.data[col].astype(str).str.replace(',', '.').astype(float)
                    except:
                        pass  # Si no se puede convertir, dejar como está
            
            # Verificar si hay columnas no numéricas y eliminarlas
            non_numeric_cols = self.data.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                self.data = self.data.drop(columns=non_numeric_cols)
                messagebox.showinfo("Información", 
                                  f"Se han eliminado las siguientes columnas no numéricas: {', '.join(non_numeric_cols)}")
            
            # Verificar si hay columnas con valores faltantes
            cols_with_na = self.data.columns[self.data.isna().any()].tolist()
            if cols_with_na:
                # Rellenar valores faltantes con la media de cada columna
                self.data = self.data.fillna(self.data.mean())
                messagebox.showinfo("Información", 
                                  f"Se han rellenado valores faltantes en las columnas: {', '.join(cols_with_na)}")
            
            # Verificar que haya suficientes datos
            if len(self.data) < 10:
                messagebox.showwarning("Advertencia", 
                                     "Se encontraron menos de 10 registros. El modelo puede no ser confiable.")
            
            # Verificar si hay una columna de tiempo
            if "Tiempo_dias" not in self.data.columns:
                # Intentar encontrar una columna que pueda ser tiempo
                time_cols = [col for col in self.data.columns if "tiempo" in col.lower() or "dia" in col.lower() or "day" in col.lower()]
                if time_cols:
                    self.data.rename(columns={time_cols[0]: "Tiempo_dias"}, inplace=True)
                else:
                    # Crear una columna de tiempo basada en el índice
                    self.data["Tiempo_dias"] = np.arange(1, len(self.data) + 1)
                    messagebox.showinfo("Información", 
                                      "No se encontró una columna de tiempo. Se ha creado una columna 'Tiempo_dias' basada en el índice.")
            
            # Ordenar datos por tiempo
            self.data = self.data.sort_values(by="Tiempo_dias")
            
            # Eliminar duplicados en Tiempo_dias
            self.data = self.data.drop_duplicates(subset=["Tiempo_dias"])
            
            # Eliminar valores atípicos en la variable objetivo
            q1 = self.data[target_column].quantile(0.25)
            q3 = self.data[target_column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filtrar valores dentro de los límites y el límite biológico
            self.data = self.data[(self.data[target_column] >= lower_bound) & 
                                 (self.data[target_column] <= upper_bound) &
                                 (self.data[target_column] <= self.LIMITE_BIOLOGICO) &
                                 (self.data[target_column] > 0)]
            
            # Establecer Tiempo_dias como índice para análisis de series temporales
            self.data = self.data.set_index("Tiempo_dias")
            
            # Asegurar que el índice está ordenado
            self.data = self.data.sort_index()
            
            # Verificar si hay huecos en la serie temporal
            if self.data.index.is_monotonic_increasing and not self.data.index.is_unique:
                messagebox.showwarning("Advertencia", 
                                     "Hay valores duplicados en la columna de tiempo. Esto puede afectar al modelo VAR.")
            
            # Verificar si hay saltos en la serie temporal
            expected_indices = np.arange(self.data.index.min(), self.data.index.max() + 1)
            missing_indices = set(expected_indices) - set(self.data.index)
            
            if missing_indices:
                messagebox.showinfo("Información", 
                                  f"Hay {len(missing_indices)} días faltantes en la serie temporal. Se realizará una interpolación.")
                
                # Crear un nuevo DataFrame con todos los índices
                full_index = pd.Index(expected_indices, name="Tiempo_dias")
                self.data = self.data.reindex(full_index)
                
                # Interpolar valores faltantes
                self.data = self.data.interpolate(method='linear')
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros válidos")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            return False
    
    def prepare_data(self):
        """Prepara los datos para el modelo VAR"""
        try:
            # Seleccionar variables para el modelo
            if self.auto_select_var.get():
                # Selección automática de variables: usar todas las numéricas
                self.model_vars = self.data.select_dtypes(include=['number']).columns.tolist()
            else:
                # Usar todas las variables
                self.model_vars = self.data.columns.tolist()
            
            # Asegurar que la variable objetivo está incluida
            target_column = self.target_var.get()
            if target_column not in self.model_vars:
                self.model_vars.append(target_column)
            
            # Crear dataset para el modelo VAR
            self.var_data = self.data[self.model_vars].copy()
            
            # Normalizar datos si está activado
            if self.normalize_var.get():
                self.var_data = pd.DataFrame(
                    self.scaler.fit_transform(self.var_data),
                    index=self.var_data.index,
                    columns=self.var_data.columns
                )
            
            # Verificar estacionariedad y aplicar diferenciación si es necesario
            self.diff_order = {}
            self.original_data = self.var_data.copy()
            
            if self.diff_var.get() == "auto":
                # Determinar automáticamente el orden de diferenciación para cada variable
                for col in self.var_data.columns:
                    # Prueba de Dickey-Fuller aumentada para verificar estacionariedad
                    adf_result = adfuller(self.var_data[col].dropna())
                    p_value = adf_result[1]
                    
                    # Si p-value > 0.05, la serie no es estacionaria
                    if p_value > 0.05:
                        # Aplicar diferenciación de primer orden
                        self.var_data[col] = self.var_data[col].diff().dropna()
                        self.diff_order[col] = 1
                        
                        # Verificar nuevamente
                        adf_result = adfuller(self.var_data[col].dropna())
                        p_value = adf_result[1]
                        
                        # Si aún no es estacionaria, aplicar diferenciación de segundo orden
                        if p_value > 0.05:
                            self.var_data[col] = self.var_data[col].diff().dropna()
                            self.diff_order[col] = 2
                    else:
                        self.diff_order[col] = 0
                
                # Eliminar filas con NaN después de la diferenciación
                self.var_data = self.var_data.dropna()
            else:
                # Aplicar el mismo orden de diferenciación a todas las variables
                diff_order = int(self.diff_var.get())
                if diff_order > 0:
                    for col in self.var_data.columns:
                        self.var_data[col] = self.var_data[col].diff(diff_order).dropna()
                        self.diff_order[col] = diff_order
                    
                    # Eliminar filas con NaN después de la diferenciación
                    self.var_data = self.var_data.dropna()
                else:
                    for col in self.var_data.columns:
                        self.diff_order[col] = 0
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar datos: {str(e)}")
            return False
    
    def select_order(self):
        """Selecciona el orden óptimo para el modelo VAR"""
        try:
            self.status_var.set("Seleccionando orden óptimo para el modelo VAR...")
            self.root.update()
            
            # Obtener el orden máximo a probar
            maxlag = int(self.maxlag_var.get())
            
            # Limitar maxlag a un valor razonable basado en los datos disponibles
            max_possible_lag = min(maxlag, len(self.var_data) // 5)
            if max_possible_lag < maxlag:
                messagebox.showinfo("Información", 
                                  f"El orden máximo se ha limitado a {max_possible_lag} debido a la cantidad de datos disponibles.")
                maxlag = max_possible_lag
            
            # Obtener el criterio de selección
            criteria = self.criteria_var.get()
            
            # Crear modelo VAR
            model = VAR(self.var_data)
            
            # Seleccionar orden óptimo
            results = model.select_order(maxlags=maxlag)
            
            # Obtener el orden óptimo según el criterio seleccionado
            if criteria == 'aic':
                selected_order = results.aic
            elif criteria == 'bic':
                selected_order = results.bic
            elif criteria == 'hqic':
                selected_order = results.hqic
            elif criteria == 'fpe':
                selected_order = results.fpe
            else:
                selected_order = results.aic  # Por defecto
            
            # Guardar resultados para mostrarlos después
            self.order_results = results
            
            # Si el orden óptimo es 0, usar 1 como mínimo
            if selected_order == 0:
                selected_order = 1
                messagebox.showinfo("Información", 
                                  "El orden óptimo seleccionado fue 0. Se usará un orden de 1 como mínimo.")
            
            return selected_order
        except Exception as e:
            messagebox.showerror("Error", f"Error al seleccionar orden: {str(e)}")
            return 1  # Valor por defecto
    
    def train_model(self):
        """Entrena el modelo VAR con los parámetros óptimos"""
        try:
            self.status_var.set("Entrenando modelo VAR...")
            self.root.update()
            
            # Seleccionar orden óptimo
            order = self.select_order()
            
            # Crear y entrenar modelo VAR
            model = VAR(self.var_data)
            self.model_fit = model.fit(order)
            
            # Guardar información del modelo
            self.model_order = order
            
            # Calcular métricas de ajuste
            self.calculate_fit_metrics()
            
            self.status_var.set(f"Modelo VAR entrenado con éxito (orden = {order})")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")
            self.status_var.set("Error al entrenar el modelo")
            return False
    
    def calculate_fit_metrics(self):
        """Calcula métricas de ajuste del modelo"""
        try:
            # Obtener la variable objetivo
            target_column = self.target_var.get()
            
            # Obtener índices de las variables en el modelo
            target_idx = list(self.var_data.columns).index(target_column)
            
            # Obtener predicciones en muestra
            in_sample_pred = self.model_fit.fittedvalues
            
            # Si se aplicó diferenciación, necesitamos transformar las predicciones
            if self.diff_order[target_column] > 0:
                # Crear una serie con las predicciones para la variable objetivo
                pred_series = pd.Series(
                    in_sample_pred[:, target_idx], 
                    index=self.var_data.index
                )
                
                # Invertir la diferenciación
                if self.diff_order[target_column] == 1:
                    # Para diferenciación de primer orden
                    # Necesitamos el primer valor original para acumular correctamente
                    original_first_idx = self.original_data.index[0]
                    if original_first_idx in self.var_data.index:
                        # Si el primer valor está en los datos diferenciados
                        original_first_value = self.original_data.loc[original_first_idx, target_column]
                    else:
                        # Si no, usar el primer valor disponible
                        original_first_value = self.original_data.iloc[0][target_column]
                    
                    # Acumular las diferencias
                    pred_cumsum = pred_series.cumsum() + original_first_value
                    self.in_sample_pred = pred_cumsum
                elif self.diff_order[target_column] == 2:
                    # Para diferenciación de segundo orden (más complejo)
                    # Necesitamos los dos primeros valores originales
                    try:
                        original_first = self.original_data.iloc[0][target_column]
                        original_second = self.original_data.iloc[1][target_column]
                        first_diff = original_second - original_first
                        
                        # Primero invertimos la segunda diferenciación
                        first_idx = self.var_data.index[0] - 1
                        temp_series = pd.Series([first_diff], index=[first_idx])
                        temp_series = pd.concat([temp_series, pred_series])
                        first_diff_restored = temp_series.cumsum()
                        
                        # Luego invertimos la primera diferenciación
                        second_idx = first_idx - 1
                        temp_series2 = pd.Series([original_first], index=[second_idx])
                        temp_series2 = pd.concat([temp_series2, first_diff_restored])
                        self.in_sample_pred = temp_series2.cumsum()[1:]
                    except:
                        # Si hay problemas con la inversión de diferenciación de segundo orden,
                        # usar un método más simple
                        self.in_sample_pred = pred_series
                        messagebox.showwarning("Advertencia", 
                                            "No se pudo invertir completamente la diferenciación de segundo orden.")
            else:
                # Si no hay diferenciación, usar directamente las predicciones
                self.in_sample_pred = pd.Series(
                    in_sample_pred[:, target_idx], 
                    index=self.var_data.index
                )
            
            # Calcular métricas solo para los índices que coinciden
            common_indices = self.in_sample_pred.index.intersection(self.original_data.index)
            
            if len(common_indices) == 0:
                # Si no hay índices en común, no podemos calcular métricas
                messagebox.showwarning("Advertencia", 
                                    "No se pudieron calcular métricas debido a problemas con los índices.")
                # Establecer valores predeterminados para evitar errores
                self.mse = 0.0
                self.rmse = 0.0
                self.mae = 0.0
                self.r2 = 0.0
                self.residuals = pd.Series([0.0])
                return False
            
            # Filtrar por índices comunes
            actual_values = self.original_data.loc[common_indices, target_column]
            predicted_values = self.in_sample_pred.loc[common_indices]
            
            # Calcular métricas
            self.mse = mean_squared_error(actual_values, predicted_values)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(actual_values, predicted_values)
            self.r2 = r2_score(actual_values, predicted_values)
            
            # Calcular residuos
            self.residuals = actual_values - predicted_values
            
            return True
        except Exception as e:
            print(f"Error al calcular métricas: {str(e)}")
            # Establecer valores predeterminados para evitar errores
            self.mse = 0.0
            self.rmse = 0.0
            self.mae = 0.0
            self.r2 = 0.0
            self.residuals = pd.Series([0.0])
            messagebox.showwarning("Advertencia", f"Error al calcular métricas: {str(e)}")
            return False
    
    def predict_future(self, days_to_predict):
        """Realiza predicciones para días futuros"""
        try:
            self.status_var.set("Prediciendo valores futuros...")
            self.root.update()
            
            # Obtener la variable objetivo
            target_column = self.target_var.get()
            target_idx = list(self.var_data.columns).index(target_column)
            
            # Realizar predicción
            forecast = self.model_fit.forecast(self.var_data.values[-self.model_order:], steps=days_to_predict)
            
            # Convertir a DataFrame
            forecast_index = np.arange(self.var_data.index[-1] + 1, self.var_data.index[-1] + days_to_predict + 1)
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=self.var_data.columns)
            
            # Si se aplicó diferenciación, necesitamos transformar las predicciones
            if self.diff_order[target_column] > 0:
                # Invertir la diferenciación
                if self.diff_order[target_column] == 1:
                    # Para diferenciación de primer orden
                    last_original_value = self.original_data.iloc[-1][target_column]
                    forecast_values = forecast_df[target_column].values
                    
                    # Calcular valores acumulativos
                    cumulative_values = np.zeros_like(forecast_values)
                    cumulative_values[0] = forecast_values[0] + last_original_value
                    
                    for i in range(1, len(forecast_values)):
                        cumulative_values[i] = cumulative_values[i-1] + forecast_values[i]
                    
                    forecast_df[target_column] = cumulative_values
                    
                elif self.diff_order[target_column] == 2:
                    # Para diferenciación de segundo orden
                    # Necesitamos los dos últimos valores originales
                    last_original = self.original_data.iloc[-1][target_column]
                    second_last_original = self.original_data.iloc[-2][target_column]
                    last_diff = last_original - second_last_original
                    
                    forecast_values = forecast_df[target_column].values
                    
                    # Primero invertimos la segunda diferenciación
                    first_diff_values = np.zeros_like(forecast_values)
                    first_diff_values[0] = forecast_values[0] + last_diff
                    
                    for i in range(1, len(forecast_values)):
                        first_diff_values[i] = first_diff_values[i-1] + forecast_values[i]
                    
                    # Luego invertimos la primera diferenciación
                    cumulative_values = np.zeros_like(first_diff_values)
                    cumulative_values[0] = first_diff_values[0] + last_original
                    
                    for i in range(1, len(first_diff_values)):
                        cumulative_values[i] = cumulative_values[i-1] + first_diff_values[i]
                    
                    forecast_df[target_column] = cumulative_values
            
            # Aplicar restricciones biológicas si está activado
            if self.bio_constraints_var.get():
                # Obtener el último valor conocido
                last_known_value = self.original_data.iloc[-1][target_column]
                
                # Asegurar que el pez no encoja
                for i in range(len(forecast_df)):
                    if i == 0:
                        forecast_df.iloc[i, target_idx] = max(forecast_df.iloc[i, target_idx], last_known_value)
                    else:
                        forecast_df.iloc[i, target_idx] = max(forecast_df.iloc[i, target_idx], forecast_df.iloc[i-1, target_idx])
                
                # Limitar al límite biológico
                forecast_df[target_column] = forecast_df[target_column].apply(lambda x: min(x, self.LIMITE_BIOLOGICO))
                
                # Desacelerar crecimiento cerca del límite biológico
                for i in range(len(forecast_df)):
                    if forecast_df.iloc[i, target_idx] > self.LIMITE_BIOLOGICO * 0.95:
                        if i == 0:
                            # Calcular crecimiento logarítmico a medida que se acerca al límite
                            distancia_al_limite = self.LIMITE_BIOLOGICO - last_known_value
                            crecimiento = distancia_al_limite * 0.05  # 5% de la distancia restante al límite
                            forecast_df.iloc[i, target_idx] = last_known_value + crecimiento
                        else:
                            distancia_al_limite = self.LIMITE_BIOLOGICO - forecast_df.iloc[i-1, target_idx]
                            crecimiento = distancia_al_limite * 0.05
                            forecast_df.iloc[i, target_idx] = forecast_df.iloc[i-1, target_idx] + crecimiento
            
            # Calcular tasas de crecimiento
            forecast_df["Growth_Rate"] = 0.0
            forecast_df["Growth_Rate_Pct"] = 0.0
            
            for i in range(len(forecast_df)):
                if i == 0:
                    prev_length = self.original_data.iloc[-1][target_column]
                else:
                    prev_length = forecast_df.iloc[i-1][target_column]
                
                current_length = forecast_df.iloc[i][target_column]
                growth = current_length - prev_length
                
                forecast_df.iloc[i, forecast_df.columns.get_loc("Growth_Rate")] = growth
                
                if prev_length > 0:
                    growth_pct = (growth / prev_length) * 100
                    forecast_df.iloc[i, forecast_df.columns.get_loc("Growth_Rate_Pct")] = growth_pct
            
            # Resetear el índice para tener Tiempo_dias como columna
            forecast_df = forecast_df.reset_index()
            forecast_df.rename(columns={"index": "Tiempo_dias"}, inplace=True)
            
            self.status_var.set("Predicción completada")
            return forecast_df
        except Exception as e:
            messagebox.showerror("Error", f"Error al predecir valores futuros: {str(e)}")
            self.status_var.set("Error en la predicción")
            return None
    
    def display_metrics(self):
        """Muestra las métricas del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.metrics_text.delete(1.0, tk.END)
        
        # Formatear métricas
        metrics_str = "MÉTRICAS DE RENDIMIENTO DEL MODELO\n"
        metrics_str += "=" * 40 + "\n\n"
        
        metrics_str += f"R² (Coeficiente de Determinación): {self.r2:.6f}\n"
        metrics_str += f"   Interpretación: {self.r2 * 100:.2f}% de la variabilidad explicada\n\n"
        
        metrics_str += f"Error Cuadrático Medio (MSE): {self.mse:.6f}\n"
        metrics_str += f"Raíz del Error Cuadrático Medio (RMSE): {self.rmse:.6f}\n"
        metrics_str += f"Error Absoluto Medio (MAE): {self.mae:.6f}\n\n"
        
        metrics_str += "Criterios de Información:\n"
        metrics_str += f"AIC: {self.model_fit.aic:.4f}\n"
        metrics_str += f"BIC: {self.model_fit.bic:.4f}\n"
        metrics_str += f"FPE: {self.model_fit.fpe:.4f}\n"
        metrics_str += f"HQIC: {self.model_fit.hqic:.4f}\n\n"
        
        metrics_str += "INTERPRETACIÓN DE MÉTRICAS\n"
        metrics_str += "=" * 40 + "\n\n"
        
        # Interpretación de R²
        if self.r2 >= 0.9:
            metrics_str += "R²: Excelente ajuste del modelo a los datos.\n"
        elif self.r2 >= 0.8:
            metrics_str += "R²: Muy buen ajuste del modelo a los datos.\n"
        elif self.r2 >= 0.7:
            metrics_str += "R²: Buen ajuste del modelo a los datos.\n"
        elif self.r2 >= 0.6:
            metrics_str += "R²: Ajuste moderado del modelo a los datos.\n"
        elif self.r2 >= 0.5:
            metrics_str += "R²: Ajuste aceptable del modelo a los datos.\n"
        else:
            metrics_str += "R²: Ajuste débil del modelo a los datos.\n"
        
        # Interpretación de errores
        metrics_str += f"Errores (MSE, RMSE, MAE): Valores más bajos indican mejor precisión.\n"
        
        # Interpretación de criterios de información
        metrics_str += "\nCriterios de Información: Valores más bajos indican mejor ajuste del modelo.\n"
        metrics_str += "Se utilizan para comparar diferentes especificaciones del modelo.\n"
        
        self.metrics_text.insert(tk.END, metrics_str)
    
    def display_parameters(self):
        """Muestra los parámetros del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.params_text.delete(1.0, tk.END)
        
        # Formatear parámetros
        params_str = "PARÁMETROS DEL MODELO VAR\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"Orden del modelo (p): {self.model_order}\n"
        params_str += f"Número de variables: {len(self.model_vars)}\n"
        params_str += f"Número de observaciones: {len(self.var_data)}\n\n"
        
        params_str += "Variables incluidas en el modelo:\n"
        for var in self.model_vars:
            diff_order = self.diff_order[var]
            params_str += f"- {var} (diferenciación: {diff_order})\n"
        
        params_str += "\nEstacionariedad de las series:\n"
        for var in self.model_vars:
            # Realizar prueba de Dickey-Fuller en los datos originales
            adf_result = adfuller(self.original_data[var].dropna())
            p_value = adf_result[1]
            
            if p_value <= 0.05:
                params_str += f"- {var}: Estacionaria (p-value: {p_value:.4f})\n"
            else:
                params_str += f"- {var}: No estacionaria (p-value: {p_value:.4f})\n"
        
        params_str += "\nResumen del modelo:\n"
        params_str += "El modelo VAR captura las relaciones dinámicas entre las variables\n"
        params_str += f"con {self.model_order} rezagos temporales.\n\n"
        
        # Añadir información sobre la causalidad de Granger si hay suficientes datos
        if len(self.var_data) > 30:  # Solo si hay suficientes datos
            params_str += "Análisis de causalidad de Granger:\n"
            target_column = self.target_var.get()
            
            for var in self.model_vars:
                if var != target_column:
                    try:
                        granger_result = self.model_fit.test_causality(target_column, var, kind='wald')
                        p_value = granger_result['pvalue']
                        
                        if p_value <= 0.05:
                            params_str += f"- {var} → {target_column}: Causalidad significativa (p-value: {p_value:.4f})\n"
                        else:
                            params_str += f"- {var} → {target_column}: No hay causalidad significativa (p-value: {p_value:.4f})\n"
                    except:
                        params_str += f"- {var} → {target_column}: No se pudo calcular la causalidad\n"
        
        self.params_text.insert(tk.END, params_str)
    
    def display_predictions_table(self, future_data):
        """Muestra la tabla de predicciones"""
        # Limpiar tabla anterior
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Limpiar estadísticas de crecimiento
        self.growth_stats_text.delete(1.0, tk.END)
        
        # Añadir datos a la tabla
        target_column = self.target_var.get()
        last_known_length = self.original_data.iloc[-1][target_column]
        
        for i, row in future_data.iterrows():
            day = int(row["Tiempo_dias"])
            length = row[target_column]
            growth = row["Growth_Rate"]
            growth_pct = row["Growth_Rate_Pct"]
            
            # Determinar el estado del crecimiento
            if length >= self.LIMITE_BIOLOGICO * 0.95:
                estado = "Cerca del límite biológico"
                tag = "limite"
            elif growth <= 0:
                estado = "Sin crecimiento"
                tag = "sin_crecimiento"
            elif growth > 0.2:
                estado = "Crecimiento rápido"
                tag = "rapido"
            elif growth < 0.05:
                estado = "Crecimiento lento"
                tag = "lento"
            else:
                estado = "Crecimiento normal"
                tag = "normal"
            
            # Insertar fila en la tabla
            item_id = self.predictions_tree.insert("", "end", values=(
                day,
                f"{length:.2f}",
                f"{growth:.4f}",
                f"{growth_pct:.2f}",
                estado
            ))
            
            # Aplicar etiqueta para colorear
            self.predictions_tree.item(item_id, tags=(tag,))
        
        # Configurar colores para los tags
        self.predictions_tree.tag_configure("limite", background="#f8d7da")
        self.predictions_tree.tag_configure("sin_crecimiento", background="#f8d7da")
        self.predictions_tree.tag_configure("rapido", background="#d4edda")
        self.predictions_tree.tag_configure("lento", background="#fff3cd")
        self.predictions_tree.tag_configure("normal", background="#e8f4f8")
        
        # Mostrar estadísticas de crecimiento
        growth_rates = future_data["Growth_Rate"]
        growth_pcts = future_data["Growth_Rate_Pct"]
        
        stats_str = "ESTADÍSTICAS DE CRECIMIENTO PREDICHO\n"
        stats_str += "=" * 40 + "\n\n"
        
        stats_str += f"Crecimiento diario promedio: {growth_rates.mean():.4f} cm/día ({growth_pcts.mean():.2f}%/día)\n"
        stats_str += f"Desviación estándar: {growth_rates.std():.4f} cm/día\n"
        stats_str += f"Crecimiento diario mínimo: {growth_rates.min():.4f} cm/día\n"
        stats_str += f"Crecimiento diario máximo: {growth_rates.max():.4f} cm/día\n"
        stats_str += f"Crecimiento total en {len(future_data)} días: {future_data[target_column].iloc[-1] - last_known_length:.2f} cm\n"
        
        self.growth_stats_text.insert(tk.END, stats_str)
    
    def display_growth_plot(self, future_data):
        """Muestra el gráfico de crecimiento"""
        # Limpiar gráficos anteriores
        for widget in self.growth_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.growth_rate_frame.winfo_children():
            widget.destroy()
        
        target_column = self.target_var.get()
        
        # Crear figura para el gráfico de crecimiento
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Graficar datos originales
        ax1.scatter(self.original_data.index, self.original_data[target_column], 
                  color='blue', label='Datos Originales', s=30, alpha=0.7)
        
        # Graficar datos futuros predichos
        ax1.scatter(future_data["Tiempo_dias"], future_data[target_column], 
                  color='red', label='Predicciones', s=30, alpha=0.7)
        
        # Conectar los puntos con líneas
        all_days = list(self.original_data.index) + list(future_data["Tiempo_dias"])
        all_lengths = list(self.original_data[target_column]) + list(future_data[target_column])
        ax1.plot(all_days, all_lengths, 'k--', alpha=0.5)
        
        # Añadir línea de límite biológico
        ax1.axhline(y=self.LIMITE_BIOLOGICO, color='red', linestyle='--', alpha=0.5, 
                   label=f'Límite Biológico ({self.LIMITE_BIOLOGICO} cm)')
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel(f'{target_column}')
        ax1.set_title('Predicción de Crecimiento - Modelo VAR')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.growth_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear figura para el gráfico de tasa de crecimiento
        fig2 = Figure(figsize=(10, 4))
        ax2 = fig2.add_subplot(111)
        
        # Graficar tasa de crecimiento
        ax2.bar(future_data["Tiempo_dias"], future_data["Growth_Rate"], 
               color='green', alpha=0.7, label='Tasa de Crecimiento')
        
        ax2.set_xlabel('Tiempo (días)')
        ax2.set_ylabel('Crecimiento (cm/día)')
        ax2.set_title('Tasa de Crecimiento Diario Predicho')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Incrustar el gráfico en la ventana tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=self.growth_rate_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_timeseries_analysis(self):
        """Muestra el análisis de series temporales"""
        # Limpiar gráficos anteriores
        for widget in self.timeseries_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.autocorr_frame.winfo_children():
            widget.destroy()
        
        # Crear figura para el gráfico de series temporales
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Graficar series temporales originales
        for col in self.original_data.columns:
            ax1.plot(self.original_data.index, self.original_data[col], label=col)
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel('Valor')
        ax1.set_title('Series Temporales Originales')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.timeseries_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear figura para el gráfico de autocorrelación
        fig2 = Figure(figsize=(10, 6))
        
        # Obtener la variable objetivo
        target_column = self.target_var.get()
        
        # Calcular y graficar la función de autocorrelación (ACF)
        ax2 = fig2.add_subplot(211)
        
        # Calcular ACF manualmente (simplificado)
        series = self.original_data[target_column]
        n = len(series)
        lags = min(40, n // 2)  # Número de rezagos a mostrar
        
        # Calcular la media de la serie
        mean = series.mean()
        
        # Calcular la varianza
        var = np.sum((series - mean) ** 2) / n
        
        # Calcular autocorrelaciones
        acf = np.zeros(lags + 1)
        for lag in range(lags + 1):
            # Calcular autocorrelación para el rezago actual
            numerator = 0
            for t in range(lag, n):
                numerator += (series.iloc[t] - mean) * (series.iloc[t - lag] - mean)
            
            acf[lag] = numerator / (n * var)
        
        # Graficar ACF
        ax2.bar(range(len(acf)), acf, width=0.3)
        ax2.axhline(y=0, linestyle='-', color='black')
        ax2.axhline(y=1.96/np.sqrt(n), linestyle='--', color='red')
        ax2.axhline(y=-1.96/np.sqrt(n), linestyle='--', color='red')
        
        ax2.set_xlabel('Rezago')
        ax2.set_ylabel('Autocorrelación')
        ax2.set_title(f'Función de Autocorrelación (ACF) para {target_column}')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Calcular y graficar la función de autocorrelación parcial (PACF)
        ax3 = fig2.add_subplot(212)
        
        # Calcular PACF usando regresión (método Yule-Walker)
        pacf = np.zeros(lags + 1)
        pacf[0] = 1.0  # La autocorrelación parcial en rezago 0 es 1
        
        # Para rezago 1, PACF = ACF
        if lags > 0:
            pacf[1] = acf[1]
        
        # Para rezagos mayores a 1, usar ecuaciones de Yule-Walker
        for lag in range(2, lags + 1):
            # Crear matriz de autocorrelaciones
            r = np.zeros((lag, lag))
            for i in range(lag):
                for j in range(lag):
                    r[i, j] = acf[abs(i - j)]
            
            # Vector de autocorrelaciones
            r_vector = acf[1:lag+1]
            
            try:
                # Resolver ecuaciones de Yule-Walker
                phi = np.linalg.solve(r, r_vector)
                pacf[lag] = phi[-1]  # El último coeficiente es la PACF
            except:
                # En caso de error, usar un valor cercano a cero
                pacf[lag] = 0.001
        
        # Graficar PACF
        ax3.bar(range(len(pacf)), pacf, width=0.3)
        ax3.axhline(y=0, linestyle='-', color='black')
        ax3.axhline(y=1.96/np.sqrt(n), linestyle='--', color='red')
        ax3.axhline(y=-1.96/np.sqrt(n), linestyle='--', color='red')
        
        ax3.set_xlabel('Rezago')
        ax3.set_ylabel('Autocorrelación Parcial')
        ax3.set_title(f'Función de Autocorrelación Parcial (PACF) para {target_column}')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        fig2.tight_layout()
        
        # Incrustar el gráfico en la ventana tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=self.autocorr_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_diagnostics_plot(self):
        """Muestra los gráficos de diagnóstico del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.diagnostics_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.info_criteria_frame.winfo_children():
            widget.destroy()
        
        try:
            # Crear figura para diagnósticos
            fig = Figure(figsize=(10, 12))
            
            # Obtener la variable objetivo
            target_column = self.target_var.get()
            
            # Gráfico de valores reales vs predichos
            ax1 = fig.add_subplot(321)
            actual_values = self.original_data.loc[self.in_sample_pred.index, target_column]
            
            ax1.scatter(actual_values, self.in_sample_pred, color='blue', alpha=0.7)
            ax1.plot([actual_values.min(), actual_values.max()], 
                   [actual_values.min(), actual_values.max()], 'r--')
            
            ax1.set_xlabel('Valores Reales')
            ax1.set_ylabel('Valores Predichos')
            ax1.set_title('Predicciones vs Valores Reales')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Gráfico de residuos vs valores predichos
            ax2 = fig.add_subplot(322)
            ax2.scatter(self.in_sample_pred, self.residuals, color='blue', alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='--')
            
            ax2.set_xlabel('Valores Predichos')
            ax2.set_ylabel('Residuos')
            ax2.set_title('Residuos vs Valores Predichos')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Histograma de residuos
            ax3 = fig.add_subplot(323)
            ax3.hist(self.residuals, bins=20, color='blue', alpha=0.7)
            
            ax3.set_xlabel('Residuos')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Histograma de Residuos')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # QQ plot de residuos
            ax4 = fig.add_subplot(324)
            
            # Ordenar residuos
            sorted_residuals = np.sort(self.residuals)
            n = len(sorted_residuals)
            
            # Calcular cuantiles teóricos de una distribución normal
            theoretical_quantiles = np.array([norm.ppf((i + 0.5) / n) for i in range(n)])
            
            # Graficar QQ plot
            ax4.scatter(theoretical_quantiles, sorted_residuals, color='blue', alpha=0.7)
            
            # Añadir línea de referencia
            min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax4.set_xlabel('Cuantiles Teóricos')
            ax4.set_ylabel('Cuantiles de Residuos')
            ax4.set_title('QQ Plot de Residuos')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Gráfico de autocorrelación de residuos
            ax5 = fig.add_subplot(325)
            
            # Calcular autocorrelación de residuos
            residuals_series = pd.Series(self.residuals)
            n_res = len(residuals_series)
            lags = min(20, n_res // 2)
            
            # Calcular la media de los residuos
            mean_res = residuals_series.mean()
            
            # Calcular la varianza
            var_res = np.sum((residuals_series - mean_res) ** 2) / n_res
            
            # Calcular autocorrelaciones
            acf_res = np.zeros(lags + 1)
            for lag in range(lags + 1):
                # Calcular autocorrelación para el rezago actual
                numerator = 0
                for t in range(lag, n_res):
                    numerator += (residuals_series.iloc[t] - mean_res) * (residuals_series.iloc[t - lag] - mean_res)
                
                acf_res[lag] = numerator / (n_res * var_res)
            
            # Graficar ACF de residuos
            ax5.bar(range(len(acf_res)), acf_res, width=0.3)
            ax5.axhline(y=0, linestyle='-', color='black')
            ax5.axhline(y=1.96/np.sqrt(n_res), linestyle='--', color='red')
            ax5.axhline(y=-1.96/np.sqrt(n_res), linestyle='--', color='red')
            
            ax5.set_xlabel('Rezago')
            ax5.set_ylabel('Autocorrelación')
            ax5.set_title('Autocorrelación de Residuos')
            ax5.grid(True, linestyle='--', alpha=0.7)
            
            # Gráfico de residuos a lo largo del tiempo
            ax6 = fig.add_subplot(326)
            ax6.plot(self.in_sample_pred.index, self.residuals, 'b-')
            ax6.axhline(y=0, color='red', linestyle='--')
            
            ax6.set_xlabel('Tiempo (días)')
            ax6.set_ylabel('Residuos')
            ax6.set_title('Residuos a lo Largo del Tiempo')
            ax6.grid(True, linestyle='--', alpha=0.7)
            
            fig.tight_layout()
            
            # Incrustar el gráfico en la ventana tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.diagnostics_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Crear tabla de criterios de información
            info_frame = tk.Frame(self.info_criteria_frame, bg="#f5f8fa")
            info_frame.pack(fill=tk.BOTH, expand=True)
            
            # Crear tabla con Treeview
            columns = ("Orden", "AIC", "BIC", "FPE", "HQIC")
            info_tree = ttk.Treeview(info_frame, columns=columns, show="headings", height=10)
            
            # Configurar encabezados
            for col in columns:
                info_tree.heading(col, text=col)
                info_tree.column(col, width=100, anchor="center")
            
            # Añadir datos a la tabla
            for i in range(len(self.order_results.aic_min_order)):
                order = i + 1
                aic = self.order_results.aic[order]
                bic = self.order_results.bic[order]
                fpe = self.order_results.fpe[order]
                hqic = self.order_results.hqic[order]
                
                # Determinar si es el orden óptimo según algún criterio
                is_optimal = False
                tag = ""
                
                if order == self.order_results.aic_min_order:
                    is_optimal = True
                    tag = "aic_optimal"
                elif order == self.order_results.bic_min_order:
                    is_optimal = True
                    tag = "bic_optimal"
                elif order == self.order_results.fpe_min_order:
                    is_optimal = True
                    tag = "fpe_optimal"
                elif order == self.order_results.hqic_min_order:
                    is_optimal = True
                    tag = "hqic_optimal"
                
                # Insertar fila en la tabla
                item_id = info_tree.insert("", "end", values=(
                    order,
                    f"{aic:.4f}",
                    f"{bic:.4f}",
                    f"{fpe:.4f}",
                    f"{hqic:.4f}"
                ))
                
                # Aplicar etiqueta para colorear si es óptimo
                if is_optimal:
                    info_tree.item(item_id, tags=(tag,))
            
            # Configurar colores para los tags
            info_tree.tag_configure("aic_optimal", background="#d4edda")
            info_tree.tag_configure("bic_optimal", background="#d1ecf1")
            info_tree.tag_configure("fpe_optimal", background="#fff3cd")
            info_tree.tag_configure("hqic_optimal", background="#f8d7da")
            
            # Añadir barra de desplazamiento
            scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=info_tree.yview)
            info_tree.configure(yscrollcommand=scrollbar.set)
            
            # Colocar tabla y scrollbar
            scrollbar.pack(side="right", fill="y")
            info_tree.pack(side="left", fill="both", expand=True)
            
            # Añadir leyenda
            legend_frame = tk.Frame(self.info_criteria_frame, bg="#f5f8fa", padx=10, pady=5)
            legend_frame.pack(fill="x")
            
            tk.Label(legend_frame, text="Leyenda:", bg="#f5f8fa", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
            
            aic_label = tk.Label(legend_frame, text="Óptimo según AIC", bg="#d4edda", padx=5, pady=2)
            aic_label.pack(side="left", padx=(0, 10))
            
            bic_label = tk.Label(legend_frame, text="Óptimo según BIC", bg="#d1ecf1", padx=5, pady=2)
            bic_label.pack(side="left", padx=(0, 10))
            
            fpe_label = tk.Label(legend_frame, text="Óptimo según FPE", bg="#fff3cd", padx=5, pady=2)
            fpe_label.pack(side="left", padx=(0, 10))
            
            hqic_label = tk.Label(legend_frame, text="Óptimo según HQIC", bg="#f8d7da", padx=5, pady=2)
            hqic_label.pack(side="left")
            
        except Exception as e:
            messagebox.showwarning("Advertencia", f"No se pudieron mostrar los diagnósticos: {str(e)}")
    
    def run_prediction(self):
        """Ejecuta el proceso completo de predicción"""
        # Cargar datos
        if not self.load_data():
            return
        
        # Preparar datos
        if not self.prepare_data():
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
        self.display_metrics()
        self.display_parameters()
        self.display_predictions_table(future_data)
        self.display_growth_plot(future_data)
        self.display_timeseries_analysis()
        self.display_diagnostics_plot()
        
        # Cambiar a la pestaña de resumen
        self.notebook.select(self.summary_tab)
        
        # Mostrar resumen de predicción
        target_column = self.target_var.get()
        last_day = future_data["Tiempo_dias"].max()
        last_length = future_data[target_column].iloc[-1]
        messagebox.showinfo("Predicción Completada", 
                           f"¡Predicción completada con éxito!\n\n"
                           f"En el día {last_day}, la longitud predicha del pez es {last_length:.4f} cm.\n\n"
                           f"Crecimiento total en {days_to_predict} días: {last_length - self.original_data[target_column].iloc[-1]:.4f} cm")
        
        self.status_var.set(f"Predicción completada. Longitud final: {last_length:.4f} cm")

if __name__ == "__main__":
    root = tk.Tk()
    app = VARFishPredictor(root)
    root.mainloop()