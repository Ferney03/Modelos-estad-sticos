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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

class FishLengthPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Holt-Winters")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.model = None
        self.model_fit = None
        self.max_day = 0
        self.seasonal_periods = 7  # Valor predeterminado para estacionalidad semanal
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
        
        tk.Label(title_frame, text="Modelo Holt-Winters con Autoajuste de Parámetros", 
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
        
        # Columna 1: Días a predecir
        col1 = tk.Frame(options_frame, bg="#f5f8fa")
        col1.pack(side="left", padx=(0, 20))
        
        tk.Label(col1, text="Días a predecir:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        self.days_var = tk.StringVar(value="30")
        tk.Entry(col1, textvariable=self.days_var, width=10).pack(anchor="w")
        
        # Columna 2: Modo de optimización
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Modo de optimización:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        self.optimization_mode_var = tk.StringVar(value="auto")
        modes = [
            ("Automático (recomendado)", "auto"),
            ("Búsqueda de cuadrícula", "grid"),
            ("Manual", "manual")
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(col2, text=text, variable=self.optimization_mode_var, value=mode).pack(anchor="w")
        
        # Columna 3: Parámetros manuales
        col3 = tk.Frame(options_frame, bg="#f5f8fa")
        col3.pack(side="left", padx=(0, 20))
        
        tk.Label(col3, text="Parámetros manuales:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        params_frame = tk.Frame(col3, bg="#f5f8fa")
        params_frame.pack(anchor="w")
        
        # Alpha (nivel)
        tk.Label(params_frame, text="α (nivel):", bg="#f5f8fa").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.alpha_var = tk.StringVar(value="0.2")
        tk.Entry(params_frame, textvariable=self.alpha_var, width=6).grid(row=0, column=1, padx=(0, 10))
        
        # Beta (tendencia)
        tk.Label(params_frame, text="β (tendencia):", bg="#f5f8fa").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self.beta_var = tk.StringVar(value="0.1")
        tk.Entry(params_frame, textvariable=self.beta_var, width=6).grid(row=1, column=1, padx=(0, 10))
        
        # Gamma (estacionalidad)
        tk.Label(params_frame, text="γ (estacionalidad):", bg="#f5f8fa").grid(row=2, column=0, sticky="w", padx=(0, 5))
        self.gamma_var = tk.StringVar(value="0.05")
        tk.Entry(params_frame, textvariable=self.gamma_var, width=6).grid(row=2, column=1, padx=(0, 10))
        
        # Phi (amortiguación)
        tk.Label(params_frame, text="φ (amortiguación):", bg="#f5f8fa").grid(row=3, column=0, sticky="w", padx=(0, 5))
        self.phi_var = tk.StringVar(value="0.9")
        tk.Entry(params_frame, textvariable=self.phi_var, width=6).grid(row=3, column=1, padx=(0, 10))
        
        # Columna 4: Opciones de estacionalidad
        col4 = tk.Frame(options_frame, bg="#f5f8fa")
        col4.pack(side="left")
        
        tk.Label(col4, text="Opciones de estacionalidad:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        seasonal_frame = tk.Frame(col4, bg="#f5f8fa")
        seasonal_frame.pack(anchor="w")
        
        # Periodo estacional
        tk.Label(seasonal_frame, text="Periodo:", bg="#f5f8fa").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.seasonal_period_var = tk.StringVar(value="auto")
        period_combo = ttk.Combobox(seasonal_frame, textvariable=self.seasonal_period_var, width=8)
        period_combo['values'] = ('auto', '4', '7', '12', '24', '30')
        period_combo.grid(row=0, column=1, padx=(0, 5))
        
        # Tipo de estacionalidad
        tk.Label(seasonal_frame, text="Tipo:", bg="#f5f8fa").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self.seasonal_type_var = tk.StringVar(value="auto")
        type_combo = ttk.Combobox(seasonal_frame, textvariable=self.seasonal_type_var, width=8)
        type_combo['values'] = ('auto', 'add', 'mul')
        type_combo.grid(row=1, column=1, padx=(0, 5))
        
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
        
        # Pestaña para componentes de la serie
        self.components_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.components_tab, text="Componentes de la Serie")
        
        # Pestaña para diagnóstico del modelo
        self.diagnostics_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.diagnostics_tab, text="Diagnóstico del Modelo")
        
        # Configurar contenido de las pestañas
        self.setup_summary_tab()
        self.setup_predictions_tab()
        self.setup_growth_tab()
        self.setup_components_tab()
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
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo Holt-Winters", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """El modelo de suavizamiento exponencial Holt-Winters es una técnica avanzada para pronóstico de series temporales que captura tres componentes clave:

1. Nivel (α): Representa el valor base de la serie y se actualiza con cada nueva observación.
2. Tendencia (β): Captura la dirección y magnitud del cambio a lo largo del tiempo.
3. Estacionalidad (γ): Modela patrones cíclicos que se repiten en intervalos regulares.

La amortiguación (φ) controla la velocidad a la que la tendencia se aplana en el futuro, evitando pronósticos explosivos.

Valores de parámetros cercanos a 1 dan más peso a las observaciones recientes, mientras que valores cercanos a 0 dan más peso al historial de la serie. El modelo selecciona automáticamente los parámetros óptimos que minimizan el error de predicción."""
        
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
    
    def setup_components_tab(self):
        """Configura la pestaña de componentes de la serie"""
        components_frame = tk.Frame(self.components_tab, bg="#f5f8fa", padx=15, pady=15)
        components_frame.pack(fill="both", expand=True)
        
        # Marco para los gráficos de componentes
        self.components_plot_frame = tk.LabelFrame(components_frame, text="Descomposición de la Serie Temporal", 
                                               font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                               padx=10, pady=10)
        self.components_plot_frame.pack(fill="both", expand=True)
    
    def setup_diagnostics_tab(self):
        """Configura la pestaña de diagnóstico del modelo"""
        diagnostics_frame = tk.Frame(self.diagnostics_tab, bg="#f5f8fa", padx=15, pady=15)
        diagnostics_frame.pack(fill="both", expand=True)
        
        # Marco para los gráficos de diagnóstico
        self.diagnostics_plot_frame = tk.LabelFrame(diagnostics_frame, text="Diagnóstico del Modelo", 
                                                font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                                padx=10, pady=10)
        self.diagnostics_plot_frame.pack(fill="both", expand=True)
    
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
            
            # Verificar si existen las columnas requeridas
            required_columns = ["Tiempo_dias", "Longitud_cm"]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                # Intentar detectar columnas automáticamente
                if len(self.data.columns) >= 2:
                    # Asumir que la primera columna es tiempo y la segunda es longitud
                    self.data.columns.values[0] = "Tiempo_dias"
                    self.data.columns.values[1] = "Longitud_cm"
                    messagebox.showinfo("Información", 
                                      "No se encontraron las columnas estándar. Se han renombrado las dos primeras columnas como 'Tiempo_dias' y 'Longitud_cm'.")
                else:
                    messagebox.showerror("Error", 
                                       f"No se encontraron las columnas requeridas: {', '.join(missing_columns)}")
                    return False
            
            # Reemplazar comas con puntos en columnas numéricas y convertir a float
            for col in self.data.columns:
                if self.data[col].dtype == object:
                    self.data[col] = self.data[col].astype(str).str.replace(',', '.').astype(float)
            
            # Ordenar datos por días para asegurar orden cronológico
            self.data = self.data.sort_values(by="Tiempo_dias")
            
            # Eliminar duplicados en Tiempo_dias
            self.data = self.data.drop_duplicates(subset=["Tiempo_dias"])
            
            # Eliminar valores atípicos en Longitud_cm
            q1 = self.data["Longitud_cm"].quantile(0.25)
            q3 = self.data["Longitud_cm"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filtrar valores dentro de los límites y el límite biológico
            self.data = self.data[(self.data["Longitud_cm"] >= lower_bound) & 
                                 (self.data["Longitud_cm"] <= upper_bound) &
                                 (self.data["Longitud_cm"] <= self.LIMITE_BIOLOGICO) &
                                 (self.data["Longitud_cm"] > 0)]
            
            # Verificar que haya suficientes datos
            if len(self.data) < 10:
                messagebox.showwarning("Advertencia", 
                                     "Se encontraron menos de 10 registros válidos. El modelo puede no ser confiable.")
            
            # Obtener el día máximo de los datos
            self.max_day = self.data["Tiempo_dias"].max()
            
            # Verificar que los datos tengan intervalos regulares
            time_diffs = self.data["Tiempo_dias"].diff().dropna()
            if len(set(time_diffs)) > 1:
                messagebox.showinfo("Información", 
                                  "Los datos no tienen intervalos de tiempo regulares. " +
                                  "Se procederá con los datos tal como están, pero considere regularizarlos para mejores resultados.")
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros válidos")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            return False
    
    def find_optimal_seasonal_period(self, y):
        """Encuentra el periodo estacional óptimo para los datos"""
        # Periodos estacionales comunes a probar
        periods_to_try = [4, 7, 12, 24, 30]
        periods_to_try = [p for p in periods_to_try if p < len(y) // 2]
        
        if not periods_to_try:
            return min(7, len(y) // 2)
        
        best_period = periods_to_try[0]
        best_aic = float('inf')
        
        for period in periods_to_try:
            try:
                # Probar modelo con este periodo
                model = ExponentialSmoothing(
                    y,
                    seasonal_periods=period,
                    trend='add',
                    seasonal='add',
                    damped=True
                ).fit(optimized=True)
                
                # Si el AIC es mejor, actualizar
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_period = period
            except:
                continue
        
        return best_period
    
    def find_optimal_seasonal_type(self, y, period):
        """Encuentra el tipo de estacionalidad óptimo para los datos"""
        try:
            # Probar modelo aditivo
            model_add = ExponentialSmoothing(
                y,
                seasonal_periods=period,
                trend='add',
                seasonal='add',
                damped=True
            ).fit(optimized=True)
            
            # Probar modelo multiplicativo
            model_mul = ExponentialSmoothing(
                y,
                seasonal_periods=period,
                trend='add',
                seasonal='mul',
                damped=True
            ).fit(optimized=True)
            
            # Comparar AIC
            if model_add.aic < model_mul.aic:
                return 'add'
            else:
                return 'mul'
        except:
            # Por defecto, usar aditivo (más estable)
            return 'add'
    
    def grid_search_parameters(self, y, period, seasonal_type):
        """Realiza una búsqueda de cuadrícula para encontrar los mejores parámetros"""
        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        phi_values = [0.8, 0.9, 0.95, 0.98]
        
        best_params = {'alpha': 0.2, 'beta': 0.1, 'gamma': 0.1, 'phi': 0.9}
        best_aic = float('inf')
        
        # Limitar la búsqueda para evitar tiempos de ejecución excesivos
        max_iterations = 50
        current_iteration = 0
        
        # Muestreo aleatorio de combinaciones de parámetros
        import random
        param_combinations = []
        
        for _ in range(max_iterations):
            alpha = random.choice(alpha_values)
            beta = random.choice(beta_values)
            gamma = random.choice(gamma_values)
            phi = random.choice(phi_values)
            param_combinations.append((alpha, beta, gamma, phi))
        
        for alpha, beta, gamma, phi in param_combinations:
            try:
                model = ExponentialSmoothing(
                    y,
                    seasonal_periods=period,
                    trend='add',
                    seasonal=seasonal_type,
                    damped=True
                ).fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma,
                    damping_trend=phi
                )
                
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_params = {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'phi': phi
                    }
                
                current_iteration += 1
                if current_iteration % 10 == 0:
                    self.status_var.set(f"Búsqueda de parámetros: {current_iteration}/{max_iterations} completados...")
                    self.root.update()
                
            except:
                continue
        
        return best_params
    
    def train_model(self):
        """Entrena el modelo Holt-Winters con los parámetros óptimos"""
        try:
            self.status_var.set("Entrenando modelo Holt-Winters...")
            self.root.update()
            
            # Preparar serie temporal
            y = self.data["Longitud_cm"].values
            
            # Determinar el periodo estacional
            if self.seasonal_period_var.get() == "auto":
                self.status_var.set("Determinando periodo estacional óptimo...")
                self.root.update()
                self.seasonal_periods = self.find_optimal_seasonal_period(y)
            else:
                try:
                    self.seasonal_periods = int(self.seasonal_period_var.get())
                    if self.seasonal_periods <= 1:
                        self.seasonal_periods = 7
                except:
                    self.seasonal_periods = 7
            
            # Verificar si hay suficientes datos para el periodo estacional
            if len(y) < self.seasonal_periods * 2:
                self.seasonal_periods = max(2, len(y) // 2)
                messagebox.showinfo("Información", 
                                  f"Periodo estacional ajustado a {self.seasonal_periods} debido a la cantidad limitada de datos.")
            
            # Determinar el tipo de estacionalidad
            if self.seasonal_type_var.get() == "auto":
                self.status_var.set("Determinando tipo de estacionalidad óptimo...")
                self.root.update()
                seasonal_type = self.find_optimal_seasonal_type(y, self.seasonal_periods)
            else:
                seasonal_type = self.seasonal_type_var.get()
            
            # Determinar los parámetros del modelo
            optimization_mode = self.optimization_mode_var.get()
            
            if optimization_mode == "auto":
                # Optimización automática de parámetros
                self.status_var.set("Optimizando parámetros automáticamente...")
                self.root.update()
                
                self.model = ExponentialSmoothing(
                    y,
                    seasonal_periods=self.seasonal_periods,
                    trend='add',
                    seasonal=seasonal_type,
                    damped=True
                )
                self.model_fit = self.model.fit(optimized=True)
                
                # Guardar los parámetros optimizados
                self.alpha = self.model_fit.params['smoothing_level']
                self.beta = self.model_fit.params['smoothing_trend']
                self.gamma = self.model_fit.params['smoothing_seasonal']
                self.phi = self.model_fit.params['damping_trend']
                
            elif optimization_mode == "grid":
                # Búsqueda de cuadrícula para parámetros óptimos
                self.status_var.set("Realizando búsqueda de cuadrícula para parámetros óptimos...")
                self.root.update()
                
                best_params = self.grid_search_parameters(y, self.seasonal_periods, seasonal_type)
                
                self.alpha = best_params['alpha']
                self.beta = best_params['beta']
                self.gamma = best_params['gamma']
                self.phi = best_params['phi']
                
                # Entrenar modelo con los mejores parámetros
                self.model = ExponentialSmoothing(
                    y,
                    seasonal_periods=self.seasonal_periods,
                    trend='add',
                    seasonal=seasonal_type,
                    damped=True
                )
                self.model_fit = self.model.fit(
                    smoothing_level=self.alpha,
                    smoothing_trend=self.beta,
                    smoothing_seasonal=self.gamma,
                    damping_trend=self.phi
                )
                
            else:
                # Usar parámetros manuales
                try:
                    self.alpha = float(self.alpha_var.get())
                    self.beta = float(self.beta_var.get())
                    self.gamma = float(self.gamma_var.get())
                    self.phi = float(self.phi_var.get())
                    
                    # Validar parámetros
                    if not (0 < self.alpha < 1 and 0 < self.beta < 1 and 0 < self.gamma < 1 and 0 < self.phi < 1):
                        messagebox.showwarning("Advertencia", 
                                             "Los parámetros deben estar entre 0 y 1. " +
                                             "Se usarán valores predeterminados.")
                        self.alpha, self.beta, self.gamma, self.phi = 0.2, 0.1, 0.1, 0.9
                except:
                    messagebox.showwarning("Advertencia", 
                                         "Parámetros inválidos. Se usarán valores predeterminados.")
                    self.alpha, self.beta, self.gamma, self.phi = 0.2, 0.1, 0.1, 0.9
                
                self.model = ExponentialSmoothing(
                    y,
                    seasonal_periods=self.seasonal_periods,
                    trend='add',
                    seasonal=seasonal_type,
                    damped=True
                )
                self.model_fit = self.model.fit(
                    smoothing_level=self.alpha,
                    smoothing_trend=self.beta,
                    smoothing_seasonal=self.gamma,
                    damping_trend=self.phi
                )
            
            # Calcular predicciones en datos de entrenamiento
            self.in_sample_pred = self.model_fit.fittedvalues
            
            # Calcular métricas en datos de entrenamiento
            self.mse = mean_squared_error(y, self.in_sample_pred)
            self.rmse = np.sqrt(self.mse)
            self.mae = mean_absolute_error(y, self.in_sample_pred)
            
            # Calcular R² manualmente ya que no es una métrica estándar para series temporales
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - self.in_sample_pred) ** 2)
            self.r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Almacenar predicciones para datos de entrenamiento
            self.data["Predicted_Length"] = self.in_sample_pred
            
            # Calcular AIC y BIC
            self.aic = self.model_fit.aic
            self.bic = self.model_fit.bic
            
            # Calcular residuos
            self.residuals = y - self.in_sample_pred
            
            self.status_var.set("Modelo Holt-Winters entrenado con éxito")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")
            self.status_var.set("Error al entrenar el modelo")
            return False
    
    def predict_future(self, days_to_predict):
        """Realiza predicciones para días futuros"""
        try:
            self.status_var.set("Prediciendo valores futuros...")
            self.root.update()
            
            # Realizar predicción con el modelo Holt-Winters
            forecast = self.model_fit.forecast(steps=days_to_predict)
            
            # Crear un dataframe para almacenar predicciones
            future_days = np.arange(self.max_day + 1, self.max_day + days_to_predict + 1)
            future_data = pd.DataFrame({
                "Tiempo_dias": future_days,
                "Longitud_cm": forecast,
                "Predicted_Length": forecast
            })
            
            # Calcular tasas de crecimiento
            future_data["Growth_Rate"] = future_data["Longitud_cm"].diff().fillna(0)
            
            # Aplicar restricciones biológicas
            last_known_length = self.data["Longitud_cm"].iloc[-1]
            
            for i in range(len(future_data)):
                # Asegurar que el pez no encoja
                if i == 0:
                    future_data.loc[future_data.index[i], "Longitud_cm"] = max(
                        future_data["Longitud_cm"].iloc[i], 
                        last_known_length
                    )
                else:
                    future_data.loc[future_data.index[i], "Longitud_cm"] = max(
                        future_data["Longitud_cm"].iloc[i], 
                        future_data["Longitud_cm"].iloc[i-1]
                    )
                
                # Limitar el crecimiento máximo
                future_data.loc[future_data.index[i], "Longitud_cm"] = min(
                    future_data["Longitud_cm"].iloc[i], 
                    self.LIMITE_BIOLOGICO
                )
                
                # Desacelerar crecimiento cerca del límite biológico
                if future_data["Longitud_cm"].iloc[i] > self.LIMITE_BIOLOGICO * 0.95:
                    if i > 0:
                        # Calcular crecimiento logarítmico a medida que se acerca al límite
                        distancia_al_limite = self.LIMITE_BIOLOGICO - future_data["Longitud_cm"].iloc[i-1]
                        crecimiento = distancia_al_limite * 0.05  # 5% de la distancia restante al límite
                        future_data.loc[future_data.index[i], "Longitud_cm"] = future_data["Longitud_cm"].iloc[i-1] + crecimiento
                
                # Actualizar la tasa de crecimiento
                if i > 0:
                    future_data.loc[future_data.index[i], "Growth_Rate"] = (
                        future_data["Longitud_cm"].iloc[i] - future_data["Longitud_cm"].iloc[i-1]
                    )
                else:
                    future_data.loc[future_data.index[i], "Growth_Rate"] = (
                        future_data["Longitud_cm"].iloc[i] - last_known_length
                    )
            
            # Calcular tasa de crecimiento porcentual
            future_data["Growth_Rate_Pct"] = 0.0
            for i in range(len(future_data)):
                if i == 0:
                    prev_length = last_known_length
                else:
                    prev_length = future_data["Longitud_cm"].iloc[i-1]
                
                if prev_length > 0:
                    future_data.loc[future_data.index[i], "Growth_Rate_Pct"] = (
                        future_data["Growth_Rate"].iloc[i] / prev_length * 100
                    )
            
            self.status_var.set("Predicción completada")
            return future_data
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
        
        metrics_str += f"Criterio de Información de Akaike (AIC): {self.aic:.6f}\n"
        metrics_str += f"Criterio de Información Bayesiano (BIC): {self.bic:.6f}\n\n"
        
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
        metrics_str += f"AIC/BIC: Valores más bajos indican mejor equilibrio entre ajuste y complejidad.\n"
        
        self.metrics_text.insert(tk.END, metrics_str)
    
    def display_parameters(self):
        """Muestra los parámetros del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.params_text.delete(1.0, tk.END)
        
        # Formatear parámetros
        params_str = "PARÁMETROS DEL MODELO HOLT-WINTERS\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"α (nivel): {self.alpha:.6f}\n"
        params_str += f"β (tendencia): {self.beta:.6f}\n"
        params_str += f"γ (estacionalidad): {self.gamma:.6f}\n"
        params_str += f"φ (amortiguación): {self.phi:.6f}\n\n"
        
        params_str += f"Periodo estacional: {self.seasonal_periods}\n"
        params_str += f"Tipo de estacionalidad: {self.model.seasonal}\n\n"
        
        params_str += "INTERPRETACIÓN DE PARÁMETROS\n"
        params_str += "=" * 40 + "\n\n"
        
        # Interpretación de alpha
        if self.alpha > 0.7:
            params_str += "α: Alto peso a observaciones recientes, respuesta rápida a cambios.\n"
        elif self.alpha > 0.3:
            params_str += "α: Equilibrio entre observaciones recientes e históricas.\n"
        else:
            params_str += "α: Mayor peso a observaciones históricas, respuesta suave a cambios.\n"
        
        # Interpretación de beta
        if self.beta > 0.7:
            params_str += "β: Adaptación rápida a cambios en la tendencia.\n"
        elif self.beta > 0.3:
            params_str += "β: Adaptación moderada a cambios en la tendencia.\n"
        else:
            params_str += "β: Adaptación lenta a cambios en la tendencia.\n"
        
        # Interpretación de gamma
        if self.gamma > 0.7:
            params_str += "γ: Alta sensibilidad a cambios en patrones estacionales.\n"
        elif self.gamma > 0.3:
            params_str += "γ: Sensibilidad moderada a cambios en patrones estacionales.\n"
        else:
            params_str += "γ: Baja sensibilidad a cambios en patrones estacionales.\n"
        
        # Interpretación de phi
        if self.phi > 0.95:
            params_str += "φ: Tendencia persistente a largo plazo.\n"
        elif self.phi > 0.9:
            params_str += "φ: Tendencia moderadamente amortiguada a largo plazo.\n"
        else:
            params_str += "φ: Tendencia fuertemente amortiguada a largo plazo.\n"
        
        self.params_text.insert(tk.END, params_str)
    
    def display_predictions_table(self, future_data):
        """Muestra la tabla de predicciones"""
        # Limpiar tabla anterior
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Limpiar estadísticas de crecimiento
        self.growth_stats_text.delete(1.0, tk.END)
        
        # Añadir datos a la tabla
        last_known_length = self.data["Longitud_cm"].iloc[-1]
        
        for i, row in future_data.iterrows():
            day = int(row["Tiempo_dias"])
            length = row["Longitud_cm"]
            growth = row["Growth_Rate"]
            growth_pct = row["Growth_Rate_Pct"]
            
            # Determinar el estado del crecimiento
            if i == 0:
                prev_length = last_known_length
            else:
                prev_length = future_data["Longitud_cm"].iloc[i-1]
            
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
        stats_str += f"Crecimiento total en {len(future_data)} días: {future_data['Longitud_cm'].iloc[-1] - last_known_length:.2f} cm\n"
        
        self.growth_stats_text.insert(tk.END, stats_str)
    
    def display_growth_plot(self, future_data):
        """Muestra el gráfico de crecimiento"""
        # Limpiar gráficos anteriores
        for widget in self.growth_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.growth_rate_frame.winfo_children():
            widget.destroy()
        
        # Crear figura para el gráfico de crecimiento
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Graficar datos originales
        ax1.scatter(self.data["Tiempo_dias"], self.data["Longitud_cm"], 
                  color='blue', label='Datos Originales', s=30, alpha=0.7)
        
        # Graficar ajuste del modelo en datos de entrenamiento
        ax1.plot(self.data["Tiempo_dias"], self.in_sample_pred, 
               color='green', label='Ajuste del Modelo', linestyle='-', alpha=0.7)
        
        # Graficar datos futuros predichos
        ax1.scatter(future_data["Tiempo_dias"], future_data["Longitud_cm"], 
                  color='red', label='Predicciones', s=30, alpha=0.7)
        
        # Conectar los puntos con líneas
        all_days = list(self.data["Tiempo_dias"]) + list(future_data["Tiempo_dias"])
        all_lengths = list(self.data["Longitud_cm"]) + list(future_data["Longitud_cm"])
        ax1.plot(all_days, all_lengths, 'k--', alpha=0.5)
        
        # Añadir línea de límite biológico
        ax1.axhline(y=self.LIMITE_BIOLOGICO, color='red', linestyle='--', alpha=0.5, 
                   label=f'Límite Biológico ({self.LIMITE_BIOLOGICO} cm)')
        
        # Añadir un recuadro ampliado para el área de predicción
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        
        # Solo crear recuadro ampliado si tenemos suficientes datos
        if len(self.data) > 10:
            # Crear ejes de recuadro ampliado
            axins = zoomed_inset_axes(ax1, zoom=2.5, loc='lower right')
            
            # Determinar región de zoom (últimos 10 puntos originales + predicciones)
            zoom_start = max(0, len(self.data) - 10)
            zoom_days = list(self.data["Tiempo_dias"].iloc[zoom_start:]) + list(future_data["Tiempo_dias"])
            zoom_lengths = list(self.data["Longitud_cm"].iloc[zoom_start:]) + list(future_data["Longitud_cm"])
            zoom_pred = list(self.in_sample_pred[zoom_start:]) + list(future_data["Longitud_cm"])
            
            # Graficar datos en recuadro ampliado
            axins.scatter(self.data["Tiempo_dias"].iloc[zoom_start:], self.data["Longitud_cm"].iloc[zoom_start:], 
                         color='blue', s=20, alpha=0.7)
            axins.plot(self.data["Tiempo_dias"].iloc[zoom_start:], self.in_sample_pred[zoom_start:], 
                      color='green', linestyle='-', alpha=0.7)
            axins.scatter(future_data["Tiempo_dias"], future_data["Longitud_cm"], 
                         color='red', s=20, alpha=0.7)
            axins.plot(zoom_days, zoom_pred, 'k--', alpha=0.5)
            
            # Establecer límites para recuadro ampliado
            x_min = min(zoom_days) - 1
            x_max = max(zoom_days) + 1
            y_min = min(min(zoom_lengths), min(zoom_pred)) - 0.5
            y_max = max(max(zoom_lengths), max(zoom_pred)) + 0.5
            axins.set_xlim(x_min, x_max)
            axins.set_ylim(y_min, y_max)
            
            # Desactivar marcas de recuadro ampliado
            axins.tick_params(labelleft=False, labelbottom=False)
            
            # Dibujar líneas conectoras entre recuadro ampliado y gráfico principal
            mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel('Longitud (cm)')
        ax1.set_title('Predicción de Longitud de Peces - Modelo Holt-Winters')
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
    
    def display_components_plot(self):
        """Muestra los gráficos de componentes de la serie temporal"""
        # Limpiar gráficos anteriores
        for widget in self.components_plot_frame.winfo_children():
            widget.destroy()
        
        try:
            # Crear figura para componentes
            fig = Figure(figsize=(10, 12))
            
            # Nivel
            ax1 = fig.add_subplot(411)
            ax1.plot(self.model_fit.level, label='Nivel')
            ax1.set_title('Componente de Nivel')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Tendencia
            ax2 = fig.add_subplot(412)
            ax2.plot(self.model_fit.trend, label='Tendencia')
            ax2.set_title('Componente de Tendencia')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            
            # Estacionalidad
            ax3 = fig.add_subplot(413)
            ax3.plot(self.model_fit.season, label='Estacionalidad')
            ax3.set_title('Componente de Estacionalidad')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
            
            # Patrón estacional
            ax4 = fig.add_subplot(414)
            seasonal_pattern = self.model_fit.season[-self.seasonal_periods:]
            ax4.bar(range(1, self.seasonal_periods + 1), seasonal_pattern, label='Patrón Estacional')
            ax4.set_title('Patrón Estacional')
            ax4.set_xlabel('Periodo')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
            
            fig.tight_layout()
            
            # Incrustar el gráfico en la ventana tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.components_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showwarning("Advertencia", f"No se pudieron mostrar los componentes: {str(e)}")
    
    def display_diagnostics_plot(self):
        """Muestra los gráficos de diagnóstico del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.diagnostics_plot_frame.winfo_children():
            widget.destroy()
        
        try:
            # Crear figura para diagnósticos
            fig = Figure(figsize=(10, 12))
            
            # Residuos vs tiempo
            ax1 = fig.add_subplot(411)
            ax1.plot(self.data["Tiempo_dias"], self.residuals, 'o-', color='blue', alpha=0.7)
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_title('Residuos vs Tiempo')
            ax1.set_xlabel('Tiempo (días)')
            ax1.set_ylabel('Residuos')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Histograma de residuos
            ax2 = fig.add_subplot(412)
            ax2.hist(self.residuals, bins=20, color='green', alpha=0.7)
            ax2.set_title('Histograma de Residuos')
            ax2.set_xlabel('Residuos')
            ax2.set_ylabel('Frecuencia')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # QQ plot de residuos
            ax3 = fig.add_subplot(413)
            from scipy import stats
            stats.probplot(self.residuals, dist="norm", plot=ax3)
            ax3.set_title('QQ Plot de Residuos')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Autocorrelación de residuos
            ax4 = fig.add_subplot(414)
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(self.residuals, lags=min(30, len(self.residuals) // 2), ax=ax4)
            ax4.set_title('Autocorrelación de Residuos')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            fig.tight_layout()
            
            # Incrustar el gráfico en la ventana tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.diagnostics_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showwarning("Advertencia", f"No se pudieron mostrar los diagnósticos: {str(e)}")
    
    def run_prediction(self):
        """Ejecuta el proceso completo de predicción"""
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
        self.display_metrics()
        self.display_parameters()
        self.display_predictions_table(future_data)
        self.display_growth_plot(future_data)
        self.display_components_plot()
        self.display_diagnostics_plot()
        
        # Cambiar a la pestaña de resumen
        self.notebook.select(self.summary_tab)
        
        # Mostrar resumen de predicción
        last_day = self.max_day + days_to_predict
        last_length = future_data.iloc[-1]["Longitud_cm"]
        messagebox.showinfo("Predicción Completada", 
                           f"¡Predicción completada con éxito!\n\n"
                           f"En el día {last_day}, la longitud predicha del pez es {last_length:.4f} cm.\n\n"
                           f"Crecimiento total en {days_to_predict} días: {last_length - self.data['Longitud_cm'].iloc[-1]:.4f} cm")
        
        self.status_var.set(f"Predicción completada. Longitud final: {last_length:.4f} cm")

if __name__ == "__main__":
    root = tk.Tk()
    app = FishLengthPredictor(root)
    root.mainloop()