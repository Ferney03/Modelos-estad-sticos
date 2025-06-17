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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class GARCHFishPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Modelo GARCH")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.returns = None  # Cambios relativos en la longitud
        self.model_params = None  # Parámetros del modelo GARCH
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
        
        tk.Label(title_frame, text="Modelo GARCH (Generalized Autoregressive Conditional Heteroskedasticity)", 
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
        
        # Columna 2: Parámetros del modelo GARCH
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Parámetros del modelo GARCH:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Orden GARCH(p,q)
        p_frame = tk.Frame(col2, bg="#f5f8fa")
        p_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(p_frame, text="Orden p (ARCH):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.p_var = tk.StringVar(value="1")
        tk.Entry(p_frame, textvariable=self.p_var, width=8).pack(side="left")
        
        q_frame = tk.Frame(col2, bg="#f5f8fa")
        q_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(q_frame, text="Orden q (GARCH):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.q_var = tk.StringVar(value="1")
        tk.Entry(q_frame, textvariable=self.q_var, width=8).pack(side="left")
        
        # Número de simulaciones
        sim_frame = tk.Frame(col2, bg="#f5f8fa")
        sim_frame.pack(anchor="w")
        
        tk.Label(sim_frame, text="Número de simulaciones:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.n_simulations_var = tk.StringVar(value="1000")
        tk.Entry(sim_frame, textvariable=self.n_simulations_var, width=8).pack(side="left")
        
        # Columna 3: Opciones avanzadas
        col3 = tk.Frame(options_frame, bg="#f5f8fa")
        col3.pack(side="left")
        
        tk.Label(col3, text="Opciones avanzadas:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Normalización de datos
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Normalizar datos", variable=self.normalize_var).pack(anchor="w")
        
        # Aplicar restricciones biológicas
        self.bio_constraints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Aplicar restricciones biológicas", variable=self.bio_constraints_var).pack(anchor="w")
        
        # Mostrar intervalos de confianza
        self.show_ci_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Mostrar intervalos de confianza", variable=self.show_ci_var).pack(anchor="w")
        
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
        
        # Pestaña para volatilidad
        self.volatility_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.volatility_tab, text="Análisis de Volatilidad")
        
        # Pestaña para diagnóstico del modelo
        self.diagnostics_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.diagnostics_tab, text="Diagnóstico del Modelo")
        
        # Configurar contenido de las pestañas
        self.setup_summary_tab()
        self.setup_predictions_tab()
        self.setup_growth_tab()
        self.setup_volatility_tab()
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
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo GARCH", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """El modelo GARCH (Generalized Autoregressive Conditional Heteroskedasticity) es una técnica estadística avanzada diseñada para modelar series temporales con volatilidad variable en el tiempo. Originalmente desarrollado para mercados financieros, puede aplicarse al crecimiento de peces donde:

1. La variabilidad en el crecimiento puede cambiar con el tiempo (heteroscedasticidad)
2. Períodos de alta variabilidad tienden a agruparse (clustering de volatilidad)
3. El crecimiento puede mostrar patrones no lineales

Un GARCH(p,q) modela la varianza condicional como función de:
- p términos ARCH (errores pasados al cuadrado)
- q términos GARCH (varianzas condicionales pasadas)

Esto permite:
- Capturar períodos de estabilidad y volatilidad en el crecimiento
- Generar intervalos de confianza más precisos para las predicciones
- Modelar el riesgo o incertidumbre en las proyecciones de crecimiento"""
        
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
        columns = ("Día", "Longitud (cm)", "Volatilidad", "Intervalo Inferior", "Intervalo Superior", "Estado")
        self.predictions_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        # Configurar encabezados y anchos de columna
        self.predictions_tree.heading("Día", text="Día")
        self.predictions_tree.column("Día", width=60, anchor="center")
        
        self.predictions_tree.heading("Longitud (cm)", text="Longitud (cm)")
        self.predictions_tree.column("Longitud (cm)", width=100, anchor="center")
        
        self.predictions_tree.heading("Volatilidad", text="Volatilidad")
        self.predictions_tree.column("Volatilidad", width=100, anchor="center")
        
        self.predictions_tree.heading("Intervalo Inferior", text="Intervalo Inferior")
        self.predictions_tree.column("Intervalo Inferior", width=120, anchor="center")
        
        self.predictions_tree.heading("Intervalo Superior", text="Intervalo Superior")
        self.predictions_tree.column("Intervalo Superior", width=120, anchor="center")
        
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
        self.growth_plot_frame = tk.LabelFrame(growth_frame, text="Gráfico de Crecimiento con Intervalos de Confianza", 
                                            font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        self.growth_plot_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de distribución de predicciones
        self.growth_dist_frame = tk.LabelFrame(growth_frame, text="Distribución de Predicciones Finales", 
                                            font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", padx=10, pady=10)
        self.growth_dist_frame.pack(fill="both", expand=True, pady=(15, 0))
    
    def setup_volatility_tab(self):
        """Configura la pestaña de análisis de volatilidad"""
        volatility_frame = tk.Frame(self.volatility_tab, bg="#f5f8fa", padx=15, pady=15)
        volatility_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de volatilidad
        self.volatility_plot_frame = tk.LabelFrame(volatility_frame, text="Volatilidad Estimada", 
                                                font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                                padx=10, pady=10)
        self.volatility_plot_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de retornos
        self.returns_plot_frame = tk.LabelFrame(volatility_frame, text="Retornos y Volatilidad", 
                                             font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                             padx=10, pady=10)
        self.returns_plot_frame.pack(fill="both", expand=True, pady=(15, 0))
    
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
            
            # Calcular retornos (cambios relativos)
            target_column = self.target_var.get()
            self.data['returns'] = self.data[target_column].pct_change() * 100
            self.data = self.data.dropna()  # Eliminar el primer registro que tendrá NaN en returns
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros válidos")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            return False
    
    def prepare_data(self):
        """Prepara los datos para el modelo GARCH"""
        try:
            # Obtener la serie de retornos
            self.returns = self.data['returns'].values
            
            # Normalizar datos si está activado
            if self.normalize_var.get():
                self.returns = self.scaler.fit_transform(self.returns.reshape(-1, 1)).ravel()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar datos: {str(e)}")
            return False
    
    def garch_likelihood(self, params, returns, p, q):
        """Función de verosimilitud negativa para el modelo GARCH"""
        omega = params[0]
        alpha = params[1:p+1]
        beta = params[p+1:p+q+1]
        
        T = len(returns)
        sigma2 = np.zeros(T)
        
        # Inicializar con la varianza incondicional
        sigma2[0] = np.var(returns)
        
        # Calcular varianzas condicionales
        for t in range(1, T):
            sigma2[t] = omega
            
            # Términos ARCH
            for i in range(min(t, p)):
                sigma2[t] += alpha[i] * returns[t-i-1]**2
            
            # Términos GARCH
            for j in range(min(t, q)):
                sigma2[t] += beta[j] * sigma2[t-j-1]
        
        # Evitar valores negativos o muy pequeños
        sigma2 = np.maximum(sigma2, 1e-6)
        
        # Calcular log-verosimilitud
        llh = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        
        return -llh  # Retornar negativo para minimización
    
    def fit_garch(self, p, q):
        """Ajusta un modelo GARCH(p,q) a los datos"""
        try:
            # Parámetros iniciales
            initial_params = np.array([0.01] + [0.1] * p + [0.8] * q)
            
            # Restricciones: todos los parámetros deben ser positivos
            bounds = [(1e-6, None) for _ in range(1 + p + q)]
            
            # Optimización
            result = minimize(
                self.garch_likelihood,
                initial_params,
                args=(self.returns, p, q),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Extraer parámetros optimizados
            omega = result.x[0]
            alpha = result.x[1:p+1]
            beta = result.x[p+1:p+q+1]
            
            # Guardar parámetros
            self.model_params = {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'p': p,
                'q': q
            }
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al ajustar el modelo GARCH: {str(e)}")
            return False
    
    def forecast_volatility(self, h, last_returns, last_sigma2):
        """Pronostica la volatilidad para h períodos adelante"""
        omega = self.model_params['omega']
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        p = self.model_params['p']
        q = self.model_params['q']
        
        # Inicializar pronósticos
        sigma2_forecast = np.zeros(h)
        
        for t in range(h):
            sigma2_t = omega
            
            # Términos ARCH
            for i in range(min(p, len(last_returns))):
                if t-i-1 < 0:
                    # Usar valores históricos
                    sigma2_t += alpha[i] * last_returns[len(last_returns)-i-1]**2
                else:
                    # Usar valor esperado de retornos futuros al cuadrado (que es la varianza)
                    sigma2_t += alpha[i] * sigma2_forecast[t-i-1]
            
            # Términos GARCH
            for j in range(min(q, len(last_sigma2))):
                if t-j-1 < 0:
                    # Usar valores históricos
                    sigma2_t += beta[j] * last_sigma2[len(last_sigma2)-j-1]
                else:
                    # Usar varianzas pronosticadas
                    sigma2_t += beta[j] * sigma2_forecast[t-j-1]
            
            sigma2_forecast[t] = sigma2_t
        
        return sigma2_forecast
    
    def simulate_paths(self, days_to_predict, n_simulations):
        """Simula múltiples trayectorias de crecimiento usando el modelo GARCH"""
        target_column = self.target_var.get()
        last_length = self.data[target_column].iloc[-1]
        
        # Calcular varianzas históricas
        omega = self.model_params['omega']
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        p = self.model_params['p']
        q = self.model_params['q']
        
        T = len(self.returns)
        hist_sigma2 = np.zeros(T)
        
        # Inicializar con la varianza incondicional
        hist_sigma2[0] = np.var(self.returns)
        
        # Calcular varianzas condicionales históricas
        for t in range(1, T):
            hist_sigma2[t] = omega
            
            # Términos ARCH
            for i in range(min(t, p)):
                hist_sigma2[t] += alpha[i] * self.returns[t-i-1]**2
            
            # Términos GARCH
            for j in range(min(t, q)):
                hist_sigma2[t] += beta[j] * hist_sigma2[t-j-1]
        
        # Pronosticar volatilidad futura
        forecast_sigma2 = self.forecast_volatility(
            days_to_predict, 
            self.returns[-p:] if p > 0 else np.array([]), 
            hist_sigma2[-q:] if q > 0 else np.array([])
        )
        
        # Inicializar matriz para almacenar simulaciones
        simulations = np.zeros((n_simulations, days_to_predict))
        
        # Simular trayectorias
        for i in range(n_simulations):
            # Inicializar con el último valor conocido
            length = last_length
            
            for t in range(days_to_predict):
                # Generar retorno aleatorio con la volatilidad pronosticada
                volatility = np.sqrt(forecast_sigma2[t])
                return_pct = np.random.normal(0, volatility)
                
                # Si los datos fueron normalizados, desnormalizar
                if self.normalize_var.get():
                    return_pct = self.scaler.inverse_transform(np.array([[return_pct]]))[0][0]
                
                # Calcular nuevo valor
                growth = length * (return_pct / 100)
                length += growth
                
                # Aplicar restricciones biológicas si está activado
                if self.bio_constraints_var.get():
                    # Asegurar que el pez no encoja
                    length = max(length, last_length)
                    
                    # Limitar al límite biológico
                    length = min(length, self.LIMITE_BIOLOGICO)
                    
                    # Desacelerar crecimiento cerca del límite biológico
                    if length > self.LIMITE_BIOLOGICO * 0.95:
                        # Calcular crecimiento logarítmico a medida que se acerca al límite
                        distancia_al_limite = self.LIMITE_BIOLOGICO - length
                        crecimiento = distancia_al_limite * 0.05  # 5% de la distancia restante al límite
                        length = length + crecimiento
                
                simulations[i, t] = length
        
        return simulations, forecast_sigma2
    
    def predict_future(self, days_to_predict, n_simulations):
        """Realiza predicciones para días futuros usando simulación Monte Carlo"""
        try:
            self.status_var.set("Prediciendo valores futuros...")
            self.root.update()
            
            # Obtener el último día de los datos
            last_day = self.data["Tiempo_dias"].max()
            
            # Crear días futuros
            future_days = np.arange(last_day + 1, last_day + days_to_predict + 1)
            
            # Simular trayectorias
            simulations, forecast_sigma2 = self.simulate_paths(days_to_predict, n_simulations)
            
            # Calcular estadísticas de las simulaciones
            mean_predictions = np.mean(simulations, axis=0)
            lower_bound = np.percentile(simulations, 5, axis=0)
            upper_bound = np.percentile(simulations, 95, axis=0)
            
            # Crear dataframe para almacenar predicciones
            future_data = pd.DataFrame({
                "Tiempo_dias": future_days,
                self.target_var.get(): mean_predictions,
                "Volatilidad": np.sqrt(forecast_sigma2),
                "Lower_CI": lower_bound,
                "Upper_CI": upper_bound
            })
            
            # Guardar simulaciones para uso posterior
            self.simulations = simulations
            
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
        
        # Calcular métricas de ajuste
        target_column = self.target_var.get()
        
        # Calcular varianzas condicionales históricas
        omega = self.model_params['omega']
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        p = self.model_params['p']
        q = self.model_params['q']
        
        T = len(self.returns)
        hist_sigma2 = np.zeros(T)
        
        # Inicializar con la varianza incondicional
        hist_sigma2[0] = np.var(self.returns)
        
        # Calcular varianzas condicionales históricas
        for t in range(1, T):
            hist_sigma2[t] = omega
            
            # Términos ARCH
            for i in range(min(t, p)):
                hist_sigma2[t] += alpha[i] * self.returns[t-i-1]**2
            
            # Términos GARCH
            for j in range(min(t, q)):
                hist_sigma2[t] += beta[j] * hist_sigma2[t-j-1]
        
        # Calcular residuos estandarizados
        std_residuals = self.returns / np.sqrt(hist_sigma2)
        
        # Calcular log-verosimilitud
        llh = -0.5 * np.sum(np.log(hist_sigma2) + self.returns**2 / hist_sigma2)
        
        # Calcular criterios de información
        n_params = 1 + p + q
        aic = -2 * llh + 2 * n_params
        bic = -2 * llh + n_params * np.log(T)
        
        # Formatear métricas
        metrics_str = "MÉTRICAS DE RENDIMIENTO DEL MODELO\n"
        metrics_str += "=" * 40 + "\n\n"
        
        metrics_str += f"Log-verosimilitud: {llh:.6f}\n\n"
        metrics_str += f"Criterio de Información de Akaike (AIC): {aic:.6f}\n"
        metrics_str += f"Criterio de Información Bayesiano (BIC): {bic:.6f}\n\n"
        
        metrics_str += "Estadísticas de Residuos Estandarizados:\n"
        metrics_str += f"Media: {np.mean(std_residuals):.6f}\n"
        metrics_str += f"Desviación Estándar: {np.std(std_residuals):.6f}\n"
        metrics_str += f"Asimetría: {np.mean(std_residuals**3):.6f}\n"
        metrics_str += f"Curtosis: {np.mean(std_residuals**4):.6f}\n\n"
        
        metrics_str += "INTERPRETACIÓN DE MÉTRICAS\n"
        metrics_str += "=" * 40 + "\n\n"
        
        # Interpretación de AIC/BIC
        metrics_str += "AIC/BIC: Valores más bajos indican mejor ajuste del modelo.\n\n"
        
        # Interpretación de residuos estandarizados
        if abs(np.mean(std_residuals)) < 0.1 and abs(np.std(std_residuals) - 1) < 0.1:
            metrics_str += "Residuos Estandarizados: Bien comportados, cercanos a una distribución normal estándar.\n"
        else:
            metrics_str += "Residuos Estandarizados: Muestran cierta desviación de la normalidad.\n"
        
        self.metrics_text.insert(tk.END, metrics_str)
    
    def display_parameters(self):
        """Muestra los parámetros del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.params_text.delete(1.0, tk.END)
        
        # Extraer parámetros
        omega = self.model_params['omega']
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        p = self.model_params['p']
        q = self.model_params['q']
        
        # Calcular persistencia
        persistence = np.sum(alpha) + np.sum(beta)
        
        # Formatear parámetros
        params_str = "PARÁMETROS DEL MODELO GARCH\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"Modelo: GARCH({p},{q})\n\n"
        
        params_str += f"Constante (omega): {omega:.6f}\n\n"
        
        params_str += "Coeficientes ARCH (alpha):\n"
        for i, a in enumerate(alpha):
            params_str += f"  alpha_{i+1}: {a:.6f}\n"
        
        params_str += "\nCoeficientes GARCH (beta):\n"
        for i, b in enumerate(beta):
            params_str += f"  beta_{i+1}: {b:.6f}\n"
        
        params_str += f"\nPersistencia: {persistence:.6f}\n"
        
        # Interpretación de persistencia
        params_str += "\nINTERPRETACIÓN DE PARÁMETROS\n"
        params_str += "=" * 40 + "\n\n"
        
        if persistence < 0.7:
            params_str += "Persistencia: Baja - La volatilidad tiende a revertir rápidamente a su nivel medio.\n"
        elif persistence < 0.9:
            params_str += "Persistencia: Moderada - La volatilidad muestra cierta memoria pero eventualmente revierte a su nivel medio.\n"
        elif persistence < 1:
            params_str += "Persistencia: Alta - La volatilidad tiene memoria larga y tarda en revertir a su nivel medio.\n"
        else:
            params_str += "Persistencia: Muy alta (≥1) - La volatilidad es no estacionaria, lo que puede indicar un modelo mal especificado.\n"
        
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
        last_known_length = self.data[target_column].iloc[-1]
        
        for i, row in future_data.iterrows():
            day = int(row["Tiempo_dias"])
            length = row[target_column]
            volatility = row["Volatilidad"]
            lower_ci = row["Lower_CI"]
            upper_ci = row["Upper_CI"]
            
            # Determinar el estado del crecimiento
            if length >= self.LIMITE_BIOLOGICO * 0.95:
                estado = "Cerca del límite biológico"
                tag = "limite"
            elif volatility > 1.5 * np.mean(future_data["Volatilidad"]):
                estado = "Alta volatilidad"
                tag = "alta_volatilidad"
            elif upper_ci - lower_ci > 2 * np.mean(future_data["Upper_CI"] - future_data["Lower_CI"]):
                estado = "Alta incertidumbre"
                tag = "alta_incertidumbre"
            else:
                estado = "Crecimiento normal"
                tag = "normal"
            
            # Insertar fila en la tabla
            item_id = self.predictions_tree.insert("", "end", values=(
                day,
                f"{length:.2f}",
                f"{volatility:.4f}",
                f"{lower_ci:.2f}",
                f"{upper_ci:.2f}",
                estado
            ))
            
            # Aplicar etiqueta para colorear
            self.predictions_tree.item(item_id, tags=(tag,))
        
        # Configurar colores para los tags
        self.predictions_tree.tag_configure("limite", background="#f8d7da")
        self.predictions_tree.tag_configure("alta_volatilidad", background="#fff3cd")
        self.predictions_tree.tag_configure("alta_incertidumbre", background="#d1ecf1")
        self.predictions_tree.tag_configure("normal", background="#e8f4f8")
        
        # Mostrar estadísticas de crecimiento
        target_column = self.target_var.get()
        growth = future_data[target_column].iloc[-1] - last_known_length
        growth_pct = (growth / last_known_length) * 100
        
        stats_str = "ESTADÍSTICAS DE CRECIMIENTO PREDICHO\n"
        stats_str += "=" * 40 + "\n\n"
        
        stats_str += f"Longitud inicial: {last_known_length:.2f} cm\n"
        stats_str += f"Longitud final predicha: {future_data[target_column].iloc[-1]:.2f} cm\n"
        stats_str += f"Crecimiento total en {len(future_data)} días: {growth:.2f} cm ({growth_pct:.2f}%)\n\n"
        
        stats_str += f"Volatilidad promedio: {future_data['Volatilidad'].mean():.4f}\n"
        stats_str += f"Amplitud promedio del intervalo de confianza: {(future_data['Upper_CI'] - future_data['Lower_CI']).mean():.2f} cm\n"
        
        self.growth_stats_text.insert(tk.END, stats_str)
    
    def display_growth_plot(self, future_data):
        """Muestra el gráfico de crecimiento con intervalos de confianza"""
        # Limpiar gráficos anteriores
        for widget in self.growth_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.growth_dist_frame.winfo_children():
            widget.destroy()
        
        target_column = self.target_var.get()
        
        # Crear figura para el gráfico de crecimiento
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Graficar datos originales
        ax1.scatter(self.data["Tiempo_dias"], self.data[target_column], 
                  color='blue', label='Datos Originales', s=30, alpha=0.7)
        
        # Graficar datos futuros predichos
        ax1.plot(future_data["Tiempo_dias"], future_data[target_column], 
               color='red', label='Predicción Media', linewidth=2)
        
        # Graficar intervalos de confianza si está activado
        if self.show_ci_var.get():
            ax1.fill_between(future_data["Tiempo_dias"], 
                           future_data["Lower_CI"], 
                           future_data["Upper_CI"], 
                           color='red', alpha=0.2, label='Intervalo de Confianza 90%')
        
        # Añadir línea de límite biológico
        ax1.axhline(y=self.LIMITE_BIOLOGICO, color='red', linestyle='--', alpha=0.5, 
                   label=f'Límite Biológico ({self.LIMITE_BIOLOGICO} cm)')
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel(f'{target_column}')
        ax1.set_title('Predicción de Crecimiento con Modelo GARCH')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.growth_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear figura para la distribución de predicciones finales
        fig2 = Figure(figsize=(10, 4))
        ax2 = fig2.add_subplot(111)
        
        # Obtener predicciones finales de todas las simulaciones
        final_predictions = self.simulations[:, -1]
        
        # Graficar histograma
        ax2.hist(final_predictions, bins=30, alpha=0.7, color='green', density=True)
        
        # Añadir líneas para percentiles
        p05 = np.percentile(final_predictions, 5)
        p50 = np.percentile(final_predictions, 50)
        p95 = np.percentile(final_predictions, 95)
        
        ax2.axvline(x=p05, color='red', linestyle='--', label=f'Percentil 5: {p05:.2f} cm')
        ax2.axvline(x=p50, color='black', linestyle='-', label=f'Mediana: {p50:.2f} cm')
        ax2.axvline(x=p95, color='blue', linestyle='--', label=f'Percentil 95: {p95:.2f} cm')
        
        ax2.set_xlabel('Longitud Final (cm)')
        ax2.set_ylabel('Densidad')
        ax2.set_title(f'Distribución de Longitudes Predichas en el Día {future_data["Tiempo_dias"].iloc[-1]}')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=self.growth_dist_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_volatility_analysis(self, future_data):
        """Muestra el análisis de volatilidad"""
        # Limpiar gráficos anteriores
        for widget in self.volatility_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.returns_plot_frame.winfo_children():
            widget.destroy()
        
        # Calcular varianzas condicionales históricas
        omega = self.model_params['omega']
        alpha = self.model_params['alpha']
        beta = self.model_params['beta']
        p = self.model_params['p']
        q = self.model_params['q']
        
        T = len(self.returns)
        hist_sigma2 = np.zeros(T)
        
        # Inicializar con la varianza incondicional
        hist_sigma2[0] = np.var(self.returns)
        
        # Calcular varianzas condicionales históricas
        for t in range(1, T):
            hist_sigma2[t] = omega
            
            # Términos ARCH
            for i in range(min(t, p)):
                hist_sigma2[t] += alpha[i] * self.returns[t-i-1]**2
            
            # Términos GARCH
            for j in range(min(t, q)):
                hist_sigma2[t] += beta[j] * hist_sigma2[t-j-1]
        
        # Convertir a volatilidad (desviación estándar)
        hist_volatility = np.sqrt(hist_sigma2)
        
        # Crear figura para el gráfico de volatilidad
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Graficar volatilidad histórica
        ax1.plot(self.data["Tiempo_dias"], hist_volatility, 
               color='blue', label='Volatilidad Histórica', linewidth=1.5)
        
        # Graficar volatilidad futura
        ax1.plot(future_data["Tiempo_dias"], future_data["Volatilidad"], 
               color='red', label='Volatilidad Pronosticada', linewidth=2)
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel('Volatilidad')
        ax1.set_title('Volatilidad Estimada del Crecimiento')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.volatility_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear figura para el gráfico de retornos
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        # Graficar retornos
        ax2.bar(self.data["Tiempo_dias"], self.returns, 
              color='green', alpha=0.6, label='Retornos Diarios')
        
        # Añadir bandas de volatilidad
        ax2.plot(self.data["Tiempo_dias"], 2*hist_volatility, 
               color='red', linestyle='--', label='Bandas de Volatilidad (±2σ)')
        ax2.plot(self.data["Tiempo_dias"], -2*hist_volatility, 
               color='red', linestyle='--')
        
        ax2.set_xlabel('Tiempo (días)')
        ax2.set_ylabel('Retorno (%)')
        ax2.set_title('Retornos Diarios y Bandas de Volatilidad')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=self.returns_plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_diagnostics_plot(self):
        """Muestra los gráficos de diagnóstico del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.diagnostics_plot_frame.winfo_children():
            widget.destroy()
        
        try:
            # Calcular varianzas condicionales históricas
            omega = self.model_params['omega']
            alpha = self.model_params['alpha']
            beta = self.model_params['beta']
            p = self.model_params['p']
            q = self.model_params['q']
            
            T = len(self.returns)
            hist_sigma2 = np.zeros(T)
            
            # Inicializar con la varianza incondicional
            hist_sigma2[0] = np.var(self.returns)
            
            # Calcular varianzas condicionales históricas
            for t in range(1, T):
                hist_sigma2[t] = omega
                
                # Términos ARCH
                for i in range(min(t, p)):
                    hist_sigma2[t] += alpha[i] * self.returns[t-i-1]**2
                
                # Términos GARCH
                for j in range(min(t, q)):
                    hist_sigma2[t] += beta[j] * hist_sigma2[t-j-1]
            
            # Calcular residuos estandarizados
            std_residuals = self.returns / np.sqrt(hist_sigma2)
            
            # Crear figura para diagnósticos
            fig = Figure(figsize=(10, 12))
            
            # QQ-plot de residuos estandarizados
            ax1 = fig.add_subplot(321)
            
            # Ordenar residuos
            sorted_residuals = np.sort(std_residuals)
            
            # Calcular cuantiles teóricos de una normal estándar
            n = len(sorted_residuals)
            theoretical_quantiles = np.array([norm.ppf((i + 0.5) / n) for i in range(n)])
            
            ax1.scatter(theoretical_quantiles, sorted_residuals, alpha=0.7)
            ax1.plot([-3, 3], [-3, 3], 'r--')
            ax1.set_xlabel('Cuantiles Teóricos')
            ax1.set_ylabel('Cuantiles Observados')
            ax1.set_title('QQ-Plot de Residuos Estandarizados')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Histograma de residuos estandarizados
            ax2 = fig.add_subplot(322)
            ax2.hist(std_residuals, bins=20, alpha=0.7, density=True)
            
            # Añadir curva de densidad normal
            x = np.linspace(-4, 4, 100)
            ax2.plot(x, norm.pdf(x), 'r-', linewidth=2)
            
            ax2.set_xlabel('Residuos Estandarizados')
            ax2.set_ylabel('Densidad')
            ax2.set_title('Histograma de Residuos Estandarizados')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # ACF de residuos estandarizados
            ax3 = fig.add_subplot(323)
            
            # Calcular autocorrelación
            lags = 20
            acf = np.zeros(lags+1)
            acf[0] = 1  # Autocorrelación en lag 0 es siempre 1
            
            for lag in range(1, lags+1):
                acf[lag] = np.corrcoef(std_residuals[lag:], std_residuals[:-lag])[0, 1]
            
            # Graficar ACF
            ax3.bar(range(lags+1), acf, width=0.3, alpha=0.7)
            
            # Añadir bandas de confianza
            ci = 1.96 / np.sqrt(len(std_residuals))
            ax3.axhline(y=0, color='k', linestyle='-')
            ax3.axhline(y=ci, color='r', linestyle='--')
            ax3.axhline(y=-ci, color='r', linestyle='--')
            
            ax3.set_xlabel('Lag')
            ax3.set_ylabel('Autocorrelación')
            ax3.set_title('ACF de Residuos Estandarizados')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # ACF de residuos estandarizados al cuadrado
            ax4 = fig.add_subplot(324)
            
            # Calcular autocorrelación de residuos al cuadrado
            squared_residuals = std_residuals**2
            acf_square = np.zeros(lags+1)
            acf_square[0] = 1
            
            for lag in range(1, lags+1):
                acf_square[lag] = np.corrcoef(squared_residuals[lag:], squared_residuals[:-lag])[0, 1]
            
            # Graficar ACF de residuos al cuadrado
            ax4.bar(range(lags+1), acf_square, width=0.3, alpha=0.7)
            
            # Añadir bandas de confianza
            ax4.axhline(y=0, color='k', linestyle='-')
            ax4.axhline(y=ci, color='r', linestyle='--')
            ax4.axhline(y=-ci, color='r', linestyle='--')
            
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelación')
            ax4.set_title('ACF de Residuos Estandarizados al Cuadrado')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Residuos vs tiempo
            ax5 = fig.add_subplot(325)
            ax5.plot(self.data["Tiempo_dias"], std_residuals, 'o-', alpha=0.7, markersize=3)
            ax5.axhline(y=0, color='k', linestyle='-')
            ax5.set_xlabel('Tiempo (días)')
            ax5.set_ylabel('Residuos Estandarizados')
            ax5.set_title('Residuos Estandarizados vs Tiempo')
            ax5.grid(True, linestyle='--', alpha=0.7)
            
            # Volatilidad vs tiempo
            ax6 = fig.add_subplot(326)
            ax6.plot(self.data["Tiempo_dias"], np.sqrt(hist_sigma2), 'o-', alpha=0.7, markersize=3)
            ax6.set_xlabel('Tiempo (días)')
            ax6.set_ylabel('Volatilidad')
            ax6.set_title('Volatilidad Estimada vs Tiempo')
            ax6.grid(True, linestyle='--', alpha=0.7)
            
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
        
        # Preparar datos
        if not self.prepare_data():
            return
        
        # Obtener parámetros del modelo GARCH
        try:
            p = int(self.p_var.get())
            q = int(self.q_var.get())
            
            if p < 0 or q < 0:
                messagebox.showerror("Error", "Los órdenes p y q deben ser enteros no negativos")
                return
                
            n_simulations = int(self.n_simulations_var.get())
            if n_simulations <= 0:
                messagebox.showerror("Error", "El número de simulaciones debe ser un entero positivo")
                return
        except ValueError:
            messagebox.showerror("Error", "Los parámetros del modelo deben ser enteros válidos")
            return
        
        # Ajustar modelo GARCH
        self.status_var.set(f"Ajustando modelo GARCH({p},{q})...")
        self.root.update()
        
        if not self.fit_garch(p, q):
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
        future_data = self.predict_future(days_to_predict, n_simulations)
        if future_data is None:
            return
        
        # Mostrar resultados
        self.display_metrics()
        self.display_parameters()
        self.display_predictions_table(future_data)
        self.display_growth_plot(future_data)
        self.display_volatility_analysis(future_data)
        self.display_diagnostics_plot()
        
        # Cambiar a la pestaña de resumen
        self.notebook.select(self.summary_tab)
        
        # Mostrar resumen de predicción
        target_column = self.target_var.get()
        last_day = future_data["Tiempo_dias"].max()
        last_length = future_data[target_column].iloc[-1]
        lower_ci = future_data["Lower_CI"].iloc[-1]
        upper_ci = future_data["Upper_CI"].iloc[-1]
        
        messagebox.showinfo("Predicción Completada", 
                           f"¡Predicción completada con éxito!\n\n"
                           f"En el día {last_day}, la longitud predicha del pez es {last_length:.2f} cm.\n\n"
                           f"Intervalo de confianza del 90%: [{lower_ci:.2f}, {upper_ci:.2f}] cm\n\n"
                           f"Crecimiento total en {days_to_predict} días: {last_length - self.data[target_column].iloc[-1]:.2f} cm")
        
        self.status_var.set(f"Predicción completada. Longitud final: {last_length:.2f} cm")

# Importar scipy.stats.norm para el QQ-plot
from scipy.stats import norm

if __name__ == "__main__":
    root = tk.Tk()
    app = GARCHFishPredictor(root)
    root.mainloop()