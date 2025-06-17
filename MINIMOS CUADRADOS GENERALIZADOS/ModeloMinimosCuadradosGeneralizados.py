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
import statsmodels.api as sm
from statsmodels.regression.linear_model import GLS, OLS
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

class GLSFishPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Regresión GLS")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.X = None  # Variables predictoras
        self.y = None  # Variable objetivo (longitud)
        self.model = None
        self.model_results = None
        self.scaler_X = None  # Se inicializará más tarde
        self.LIMITE_BIOLOGICO = 70  # Límite biológico en cm para truchas
        
        # Parámetros automáticos
        self.best_corr_structure = None
        self.best_order = None
        
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
        
        tk.Label(title_frame, text="Modelo de Regresión de Mínimos Cuadrados Generalizados (GLS)", 
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
        
        # Columna 2: Parámetros del modelo GLS
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Parámetros del modelo GLS:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Selección automática de parámetros
        self.auto_params_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col2, text="Selección automática de parámetros", 
                      variable=self.auto_params_var, 
                      command=self.toggle_params_selection).pack(anchor="w", pady=(0, 5))
        
        # Estructura de correlación (desactivada por defecto)
        self.corr_frame = tk.Frame(col2, bg="#f5f8fa")
        self.corr_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(self.corr_frame, text="Estructura de correlación:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.corr_structure_var = tk.StringVar(value="AR")
        self.corr_combo = ttk.Combobox(self.corr_frame, textvariable=self.corr_structure_var, width=10, state="disabled")
        self.corr_combo['values'] = ('AR', 'MA', 'ARMA', 'None')
        self.corr_combo.pack(side="left")
        
        # Orden de la estructura (desactivada por defecto)
        self.order_frame = tk.Frame(col2, bg="#f5f8fa")
        self.order_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(self.order_frame, text="Orden de la estructura:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.order_var = tk.StringVar(value="1")
        self.order_entry = tk.Entry(self.order_frame, textvariable=self.order_var, width=8, state="disabled")
        self.order_entry.pack(side="left")
        
        # Tamaño del conjunto de prueba
        test_frame = tk.Frame(col2, bg="#f5f8fa")
        test_frame.pack(anchor="w")
        
        tk.Label(test_frame, text="Tamaño del conjunto de prueba (%):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.test_size_var = tk.StringVar(value="20")
        tk.Entry(test_frame, textvariable=self.test_size_var, width=8).pack(side="left")
        
        # Columna 3: Opciones avanzadas
        col3 = tk.Frame(options_frame, bg="#f5f8fa")
        col3.pack(side="left")
        
        tk.Label(col3, text="Opciones avanzadas:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Normalización de datos
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Normalizar datos", variable=self.normalize_var).pack(anchor="w")
        
        # Incluir término constante
        self.include_const_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Incluir término constante", variable=self.include_const_var).pack(anchor="w")
        
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
        
        # Pestaña para coeficientes del modelo
        self.coef_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.coef_tab, text="Coeficientes del Modelo")
        
        # Pestaña para diagnóstico del modelo
        self.diagnostics_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.diagnostics_tab, text="Diagnóstico del Modelo")
        
        # Configurar contenido de las pestañas
        self.setup_summary_tab()
        self.setup_predictions_tab()
        self.setup_growth_tab()
        self.setup_coef_tab()
        self.setup_diagnostics_tab()
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Listo para comenzar. Seleccione un archivo Excel y configure los parámetros.")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                               anchor=tk.W, font=("Arial", 9), bg="#e1e8ed", fg="#34495e", padx=10, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_params_selection(self):
        """Activa o desactiva los controles de selección manual de parámetros"""
        if self.auto_params_var.get():
            # Desactivar controles manuales
            self.corr_combo.config(state="disabled")
            self.order_entry.config(state="disabled")
        else:
            # Activar controles manuales
            self.corr_combo.config(state="normal")
            self.order_entry.config(state="normal")
    
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
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo GLS", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """La Regresión de Mínimos Cuadrados Generalizados (GLS) es una extensión de la regresión lineal ordinaria que permite manejar la autocorrelación y la heteroscedasticidad en los datos. Esto es especialmente útil para datos de series temporales como el crecimiento de peces.

Características principales del modelo GLS:

1. Maneja la autocorrelación temporal: Reconoce que las observaciones cercanas en el tiempo están relacionadas
2. Corrige la heteroscedasticidad: Permite que la varianza de los errores no sea constante
3. Proporciona estimaciones más precisas: Los coeficientes y errores estándar son más confiables que en la regresión ordinaria
4. Permite diferentes estructuras de correlación: Autoregresiva (AR), Media Móvil (MA) o combinaciones (ARMA)

Este modelo es ideal para predecir el crecimiento de peces, ya que las tasas de crecimiento suelen mostrar patrones temporales y la variabilidad puede cambiar a lo largo del tiempo."""
        
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
    
    def setup_coef_tab(self):
        """Configura la pestaña de coeficientes del modelo"""
        coef_frame = tk.Frame(self.coef_tab, bg="#f5f8fa", padx=15, pady=15)
        coef_frame.pack(fill="both", expand=True)
        
        # Marco para la tabla de coeficientes
        self.coef_table_frame = tk.LabelFrame(coef_frame, text="Coeficientes del Modelo", 
                                           font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                           padx=10, pady=10)
        self.coef_table_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de coeficientes
        self.coef_plot_frame = tk.LabelFrame(coef_frame, text="Visualización de Coeficientes", 
                                          font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                          padx=10, pady=10)
        self.coef_plot_frame.pack(fill="both", expand=True, pady=(15, 0))
    
    def setup_diagnostics_tab(self):
        """Configura la pestaña de diagnóstico del modelo"""
        diagnostics_frame = tk.Frame(self.diagnostics_tab, bg="#f5f8fa", padx=15, pady=15)
        diagnostics_frame.pack(fill="both", expand=True)
        
        # Marco para los gráficos de diagnóstico
        self.diagnostics_plot_frame = tk.LabelFrame(diagnostics_frame, text="Diagnóstico del Modelo", 
                                                font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                                padx=10, pady=10)
        self.diagnostics_plot_frame.pack(fill="both", expand=True)
        
        # Marco para pruebas de diagnóstico
        self.diagnostics_tests_frame = tk.LabelFrame(diagnostics_frame, text="Pruebas de Diagnóstico", 
                                                  font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                                  padx=10, pady=10)
        self.diagnostics_tests_frame.pack(fill="both", expand=True, pady=(15, 0))
    
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
            
            # Cargar datos - Usar engine='openpyxl' para asegurar compatibilidad con archivos .xlsx
            try:
                self.data = pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                # Si falla con openpyxl, intentar con xlrd para archivos .xls
                try:
                    self.data = pd.read_excel(file_path, engine='xlrd')
                except Exception as e2:
                    # Último intento con el motor por defecto
                    self.data = pd.read_excel(file_path)
            
            # Verificar si el DataFrame está vacío
            if self.data.empty:
                messagebox.showerror("Error", "El archivo Excel está vacío o no se pudo leer correctamente")
                return False
                
            # Imprimir información para depuración
            print(f"Archivo cargado: {file_path}")
            print(f"Columnas encontradas: {self.data.columns.tolist()}")
            print(f"Primeras filas:\n{self.data.head()}")
            
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
            
            # Calcular tasas de crecimiento
            # Corrected or removed the invalid line
            # Ensure this line is properly implemented or removed if unnecessary
            
            # Calcular tasas de crecimiento
            self.data["Growth_Rate"] = self.data[target_column].diff().fillna(0)
            
            # Calcular tasa de crecimiento porcentual
            self.data["Growth_Rate_Pct"] = 0.0
            for i in range(1, len(self.data)):
                prev_length = self.data[target_column].iloc[i-1]
                if prev_length > 0:
                    self.data.loc[self.data.index[i], "Growth_Rate_Pct"] = (
                        self.data["Growth_Rate"].iloc[i] / prev_length * 100
                    )
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros válidos")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            print(f"Error detallado: {str(e)}")  # Imprimir error detallado para depuración
            return False
    
    def prepare_data(self):
        """Prepara los datos para el modelo GLS"""
        try:
            target_column = self.target_var.get()
            
            # Separar variables predictoras y objetivo
            self.y = self.data[target_column].values
            
            # Excluir la variable objetivo y otras columnas derivadas
            exclude_cols = [target_column, "Growth_Rate", "Growth_Rate_Pct"]
            X_data = self.data.drop(columns=exclude_cols)
            
            # Añadir términos polinómicos para Tiempo_dias si existe
            if "Tiempo_dias" in X_data.columns:
                X_data["Tiempo_dias_sq"] = X_data["Tiempo_dias"] ** 2
            
            # Guardar nombres de columnas para interpretación
            self.feature_names = X_data.columns.tolist()
            
            # Convertir a matriz numpy
            self.X = X_data.values
            
            # Inicializar un scaler para cada columna
            if self.normalize_var.get():
                self.scaler_X = StandardScaler()
                self.X = self.scaler_X.fit_transform(self.X)
                
                # Normalizar la variable objetivo por separado
                self.scaler_y = StandardScaler()
                self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1)).ravel()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar datos: {str(e)}")
            return False
    
    def select_best_model(self, X_train, y_train):
        """Selecciona automáticamente la mejor estructura de correlación y orden"""
        self.status_var.set("Seleccionando automáticamente los mejores parámetros...")
        self.root.update()
        
        # Estructuras de correlación a probar
        structures = ['AR', 'MA', 'ARMA', 'None']
        max_order = min(3, len(y_train) // 10)  # Limitar el orden máximo
        
        best_aic = float('inf')
        best_structure = 'None'
        best_order = 0
        best_model = None
        best_results = None
        
        # Primero probar sin estructura de correlación (OLS)
        X_train_const = sm.add_constant(X_train) if self.include_const_var.get() else X_train
        ols_model = sm.OLS(y_train, X_train_const)
        ols_results = ols_model.fit()
        best_aic = ols_results.aic
        best_model = ols_model
        best_results = ols_results
        
        # Probar diferentes estructuras y órdenes
        for structure in structures:
            if structure == 'None':
                continue  # Ya probamos sin estructura
                
            for order in range(1, max_order + 1):
                try:
                    # Crear matriz de covarianza
                    cov = self.create_correlation_matrix(structure, order, len(y_train))
                    
                    if cov is not None:
                        # Entrenar modelo GLS
                        model = GLS(y_train, X_train_const, sigma=cov)
                        results = model.fit()
                        
                        # Comparar AIC
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_structure = structure
                            best_order = order
                            best_model = model
                            best_results = results
                except Exception as e:
                    continue  # Ignorar errores y probar la siguiente combinación
        
        # Guardar los mejores parámetros
        self.best_corr_structure = best_structure
        self.best_order = best_order
        
        # Actualizar la interfaz con los mejores parámetros
        self.corr_structure_var.set(best_structure)
        self.order_var.set(str(best_order))
        
        self.status_var.set(f"Mejor estructura seleccionada: {best_structure}, Orden: {best_order}")
        
        return best_model, best_results
    
    def create_correlation_matrix(self, structure, order, n):
        """Crea una matriz de covarianza basada en la estructura y orden especificados"""
        try:
            if structure == 'None':
                return None
            
            if structure == 'AR':
                # Calcular autocorrelaciones
                acf_values = acf(self.y, nlags=order, fft=False)
                rho = acf_values[1:order+1]  # Excluir lag 0
                
                # Crear matriz de covarianza AR
                cov = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            cov[i, j] = 1.0
                        else:
                            lag = abs(i - j)
                            if lag <= order:
                                cov[i, j] = rho[lag-1]
                            else:
                                # Para lags mayores que el orden, usar el último valor de rho
                                cov[i, j] = rho[-1] ** (lag - order + 1)
                
                return cov
            
            elif structure == 'MA':
                # Crear matriz de covarianza MA
                cov = np.zeros((n, n))
                # Estimar theta basado en autocorrelaciones
                acf_values = acf(self.y, nlags=order+1, fft=False)
                theta = acf_values[1:order+1] / acf_values[0]
                
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            # Varianza en la diagonal
                            cov[i, j] = 1.0 + sum(theta**2)
                        elif abs(i - j) <= order:
                            # Covarianza para lags dentro del orden
                            lag = abs(i - j)
                            cov[i, j] = theta[lag-1]
                        else:
                            # Covarianza cero para lags mayores que el orden
                            cov[i, j] = 0.0
                
                return cov
            
            elif structure == 'ARMA':
                # Simplificación: usar una combinación de AR y MA
                ar_cov = self.create_correlation_matrix('AR', order, n)
                ma_cov = self.create_correlation_matrix('MA', order, n)
                
                if ar_cov is not None and ma_cov is not None:
                    # Combinar las matrices (promedio ponderado)
                    cov = 0.7 * ar_cov + 0.3 * ma_cov
                    return cov
                else:
                    return None
            
            else:
                return None
            
        except Exception as e:
            print(f"Error al crear matriz de correlación: {str(e)}")
            return None
    
    def train_model(self):
        """Entrena el modelo GLS"""
        try:
            self.status_var.set("Entrenando modelo GLS...")
            self.root.update()
            
            # Dividir datos en entrenamiento y prueba
            test_size = float(self.test_size_var.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42)
            
            # Añadir constante si está activado
            if self.include_const_var.get():
                X_train = sm.add_constant(X_train)
            
            # Selección automática o manual de parámetros
            if self.auto_params_var.get():
                # Selección automática
                self.model, self.model_results = self.select_best_model(X_train, y_train)
            else:
                # Selección manual
                # Crear estructura de correlación
                corr_structure = self.corr_structure_var.get()
                order = int(self.order_var.get())
                
                if corr_structure != 'None':
                    cov = self.create_correlation_matrix(corr_structure, order, len(y_train))
                    
                    # Entrenar modelo GLS con la estructura especificada
                    self.model = GLS(y_train, X_train, sigma=cov)
                    self.model_results = self.model.fit()
                else:
                    # Usar OLS si no se especifica estructura de correlación
                    self.model = sm.OLS(y_train, X_train)
                    self.model_results = self.model.fit()
            
            # Guardar datos de entrenamiento y prueba
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            
            # Calcular predicciones en datos de entrenamiento
            self.y_train_pred = self.model_results.predict(X_train)
            
            # Añadir constante a X_test si es necesario
            if self.include_const_var.get():
                X_test = sm.add_constant(X_test)
                # Asegurar que X_test tiene la misma estructura que X_train
                if X_test.shape[1] != X_train.shape[1]:
                    # Añadir columnas faltantes con ceros
                    missing_cols = X_train.shape[1] - X_test.shape[1]
                    X_test = np.hstack([X_test, np.zeros((X_test.shape[0], missing_cols))])
            
            # Calcular predicciones en datos de prueba
            self.y_test_pred = self.model_results.predict(X_test)
            
            # Si los datos fueron normalizados, desnormalizar las predicciones
            if self.normalize_var.get():
                self.y_train_pred = self.scaler_y.inverse_transform(self.y_train_pred.reshape(-1, 1)).ravel()
                self.y_test_pred = self.scaler_y.inverse_transform(self.y_test_pred.reshape(-1, 1)).ravel()
                self.y_train = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1)).ravel()
                self.y_test = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
            
            # Calcular métricas en datos de entrenamiento
            self.train_mse = mean_squared_error(self.y_train, self.y_train_pred)
            self.train_rmse = np.sqrt(self.train_mse)
            self.train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
            self.train_r2 = r2_score(self.y_train, self.y_train_pred)
            
            # Calcular métricas en datos de prueba
            self.test_mse = mean_squared_error(self.y_test, self.y_test_pred)
            self.test_rmse = np.sqrt(self.test_mse)
            self.test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
            self.test_r2 = r2_score(self.y_test, self.y_test_pred)
            
            # Calcular residuos
            self.train_residuals = self.y_train - self.y_train_pred
            self.test_residuals = self.y_test - self.y_test_pred
            
            self.status_var.set("Modelo GLS entrenado con éxito")
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
            
            # Obtener el último día de los datos
            last_day = self.data["Tiempo_dias"].max()
            
            # Crear días futuros
            future_days = np.arange(last_day + 1, last_day + days_to_predict + 1)
            
            # Crear dataframe para almacenar predicciones
            future_data = pd.DataFrame({"Tiempo_dias": future_days})
            
            # Obtener el último registro para usar como base para la predicción
            last_record = self.data.iloc[-1].copy()
            target_column = self.target_var.get()
            
            # Inicializar predicciones
            predictions = []
            growth_rates = []
            
            # Realizar predicciones iterativas
            for i in range(days_to_predict):
                # Crear un registro para predecir
                # Usar los mismos features que se usaron en el entrenamiento
                X_pred_df = pd.DataFrame(columns=self.feature_names)
                X_pred_df.loc[0] = 0  # Inicializar con ceros
                
                # Llenar con los valores del último registro conocido
                for feature in self.feature_names:
                    if feature == "Tiempo_dias":
                        X_pred_df.loc[0, feature] = future_days[i]
                    elif feature == "Tiempo_dias_sq":
                        X_pred_df.loc[0, feature] = future_days[i] ** 2
                    elif feature in last_record:
                        X_pred_df.loc[0, feature] = last_record[feature]
                
                # Convertir a numpy array
                X_pred = X_pred_df.values
                
                # Normalizar si es necesario
                if self.normalize_var.get():
                    X_pred = self.scaler_X.transform(X_pred)
                
                # Añadir constante si es necesario
                if self.include_const_var.get():
                    X_pred = sm.add_constant(X_pred)
                    # Asegurar que X_pred tiene la misma estructura que X_train
                    if X_pred.shape[1] != self.X_train.shape[1]:
                        # Añadir columnas faltantes con ceros
                        missing_cols = self.X_train.shape[1] - X_pred.shape[1]
                        X_pred = np.hstack([X_pred, np.zeros((X_pred.shape[0], missing_cols))])
                
                # Realizar predicción
                y_pred = self.model_results.predict(X_pred)[0]
                
                # Desnormalizar si es necesario
                if self.normalize_var.get():
                    y_pred = self.scaler_y.inverse_transform(np.array([[y_pred]]))[0][0]
                
                # Aplicar restricciones biológicas si está activado
                if self.bio_constraints_var.get():
                    # Asegurar que el pez no encoja
                    if i > 0:
                        y_pred = max(y_pred, predictions[i-1])
                    else:
                        y_pred = max(y_pred, last_record[target_column])
                    
                    # Limitar al límite biológico
                    y_pred = min(y_pred, self.LIMITE_BIOLOGICO)
                    
                    # Desacelerar crecimiento cerca del límite biológico
                    if y_pred > self.LIMITE_BIOLOGICO * 0.95:
                        if i > 0:
                            # Calcular crecimiento logarítmico a medida que se acerca al límite
                            distancia_al_limite = self.LIMITE_BIOLOGICO - predictions[i-1]
                            crecimiento = distancia_al_limite * 0.05  # 5% de la distancia restante al límite
                            y_pred = predictions[i-1] + crecimiento
                        else:
                            distancia_al_limite = self.LIMITE_BIOLOGICO - last_record[target_column]
                            crecimiento = distancia_al_limite * 0.05
                            y_pred = last_record[target_column] + crecimiento
                
                # Calcular tasa de crecimiento
                if i == 0:
                    prev_length = last_record[target_column]
                else:
                    prev_length = predictions[i-1]
                
                growth_rate = y_pred - prev_length
                
                # Guardar predicción
                predictions.append(y_pred)
                growth_rates.append(growth_rate)
                
                # Actualizar el último registro con la predicción
                last_record[target_column] = y_pred
            
            # Añadir predicciones al dataframe
            future_data[target_column] = predictions
            future_data["Growth_Rate"] = growth_rates
            
            # Calcular tasa de crecimiento porcentual
            future_data["Growth_Rate_Pct"] = 0.0
            for i in range(len(future_data)):
                if i == 0:
                    prev_length = self.data[target_column].iloc[-1]
                else:
                    prev_length = future_data[target_column].iloc[i-1]
                
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
        
        metrics_str += "Conjunto de Entrenamiento:\n"
        metrics_str += f"R² (Coeficiente de Determinación): {self.train_r2:.6f}\n"
        metrics_str += f"   Interpretación: {self.train_r2 * 100:.2f}% de la variabilidad explicada\n\n"
        
        metrics_str += f"Error Cuadrático Medio (MSE): {self.train_mse:.6f}\n"
        metrics_str += f"Raíz del Error Cuadrático Medio (RMSE): {self.train_rmse:.6f}\n"
        metrics_str += f"Error Absoluto Medio (MAE): {self.train_mae:.6f}\n\n"
        
        metrics_str += "Conjunto de Prueba:\n"
        metrics_str += f"R² (Coeficiente de Determinación): {self.test_r2:.6f}\n"
        metrics_str += f"   Interpretación: {self.test_r2 * 100:.2f}% de la variabilidad explicada\n\n"
        
        metrics_str += f"Error Cuadrático Medio (MSE): {self.test_mse:.6f}\n"
        metrics_str += f"Raíz del Error Cuadrático Medio (RMSE): {self.test_rmse:.6f}\n"
        metrics_str += f"Error Absoluto Medio (MAE): {self.test_mae:.6f}\n\n"
        
        metrics_str += "Estadísticas del Modelo GLS:\n"
        metrics_str += f"Log-Likelihood: {self.model_results.llf:.4f}\n"
        metrics_str += f"AIC: {self.model_results.aic:.4f}\n"
        metrics_str += f"BIC: {self.model_results.bic:.4f}\n\n"
        
        metrics_str += "INTERPRETACIÓN DE MÉTRICAS\n"
        metrics_str += "=" * 40 + "\n\n"
        
        # Interpretación de R²
        if self.test_r2 >= 0.9:
            metrics_str += "R²: Excelente ajuste del modelo a los datos.\n"
        elif self.test_r2 >= 0.8:
            metrics_str += "R²: Muy buen ajuste del modelo a los datos.\n"
        elif self.test_r2 >= 0.7:
            metrics_str += "R²: Buen ajuste del modelo a los datos.\n"
        elif self.test_r2 >= 0.6:
            metrics_str += "R²: Ajuste moderado del modelo a los datos.\n"
        elif self.test_r2 >= 0.5:
            metrics_str += "R²: Ajuste aceptable del modelo a los datos.\n"
        else:
            metrics_str += "R²: Ajuste débil del modelo a los datos.\n"
        
        # Interpretación de errores
        metrics_str += f"Errores (MSE, RMSE, MAE): Valores más bajos indican mejor precisión.\n"
        
        # Comparación entre entrenamiento y prueba
        if self.train_r2 - self.test_r2 > 0.2:
            metrics_str += "\nAdvertencia: La diferencia entre R² de entrenamiento y prueba sugiere sobreajuste.\n"
        
        # Interpretación de AIC/BIC
        metrics_str += "\nAIC/BIC: Valores más bajos indican mejor ajuste considerando la complejidad del modelo.\n"
        
        self.metrics_text.insert(tk.END, metrics_str)
    
    def display_parameters(self):
        """Muestra los parámetros del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.params_text.delete(1.0, tk.END)
        
        # Formatear parámetros
        params_str = "PARÁMETROS DEL MODELO GLS\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"Número de variables predictoras: {len(self.feature_names)}\n"
        params_str += f"Tamaño del conjunto de entrenamiento: {len(self.y_train)} registros\n"
        params_str += f"Tamaño del conjunto de prueba: {len(self.y_test)} registros\n\n"
        
        # Mostrar estructura de correlación seleccionada
        if self.auto_params_var.get():
            params_str += f"Selección automática de parámetros: Sí\n"
            params_str += f"Estructura de correlación seleccionada: {self.best_corr_structure}\n"
            params_str += f"Orden seleccionado: {self.best_order}\n\n"
        else:
            params_str += f"Selección automática de parámetros: No\n"
            params_str += f"Estructura de correlación: {self.corr_structure_var.get()}\n"
            params_str += f"Orden de la estructura: {self.order_var.get()}\n\n"
        
        params_str += "Variables predictoras utilizadas:\n"
        for i, feature in enumerate(self.feature_names):
            params_str += f"- {feature}\n"
        
        params_str += "\nResumen del modelo:\n"
        params_str += str(self.model_results.summary().tables[0])
        
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
        ax1.scatter(self.data["Tiempo_dias"], self.data[target_column], 
                  color='blue', label='Datos Originales', s=30, alpha=0.7)
        
        # Graficar datos futuros predichos
        ax1.scatter(future_data["Tiempo_dias"], future_data[target_column], 
                  color='red', label='Predicciones', s=30, alpha=0.7)
        
        # Conectar los puntos con líneas
        all_days = list(self.data["Tiempo_dias"]) + list(future_data["Tiempo_dias"])
        all_lengths = list(self.data[target_column]) + list(future_data[target_column])
        ax1.plot(all_days, all_lengths, 'k--', alpha=0.5)
        
        # Añadir línea de límite biológico
        ax1.axhline(y=self.LIMITE_BIOLOGICO, color='red', linestyle='--', alpha=0.5, 
                   label=f'Límite Biológico ({self.LIMITE_BIOLOGICO} cm)')
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel(f'{target_column}')
        ax1.set_title('Predicción de Crecimiento - Modelo GLS')
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
    
    def display_coefficients(self):
        """Muestra los coeficientes del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.coef_table_frame.winfo_children():
            widget.destroy()
        
        for widget in self.coef_plot_frame.winfo_children():
            widget.destroy()
        
        # Crear tabla de coeficientes
        coef_frame_inner = tk.Frame(self.coef_table_frame, bg="#f5f8fa")
        coef_frame_inner.pack(fill=tk.BOTH, expand=True)
        
        # Crear tabla con Treeview
        columns = ("Variable", "Coeficiente", "Error Estándar", "t-valor", "p-valor", "Significancia")
        coef_tree = ttk.Treeview(coef_frame_inner, columns=columns, show="headings", height=15)
        
        # Configurar encabezados
        for col in columns:
            coef_tree.heading(col, text=col)
            if col == "Variable":
                coef_tree.column(col, width=150, anchor="w")
            else:
                coef_tree.column(col, width=100, anchor="center")
        
        # Obtener coeficientes del modelo
        params = self.model_results.params
        bse = self.model_results.bse
        tvalues = self.model_results.tvalues
        pvalues = self.model_results.pvalues
        
        # Nombres de las variables (incluyendo constante si está presente)
        var_names = ["const"] + self.feature_names if self.include_const_var.get() else self.feature_names
        
        # Añadir datos a la tabla
        for i, var in enumerate(var_names):
            if i < len(params):
                # Determinar significancia
                if pvalues[i] < 0.001:
                    significancia = "***"
                    tag = "alta"
                elif pvalues[i] < 0.01:
                    significancia = "**"
                    tag = "alta"
                elif pvalues[i] < 0.05:
                    significancia = "*"
                    tag = "media"
                elif pvalues[i] < 0.1:
                    significancia = "."
                    tag = "media"
                else:
                    significancia = ""
                    tag = "baja"
                
                # Insertar fila en la tabla
                item_id = coef_tree.insert("", "end", values=(
                    var,
                    f"{params[i]:.6f}",
                    f"{bse[i]:.6f}",
                    f"{tvalues[i]:.4f}",
                    f"{pvalues[i]:.4f}",
                    significancia
                ))
                
                # Aplicar etiqueta para colorear
                coef_tree.item(item_id, tags=(tag,))
        
        # Configurar colores para los tags
        coef_tree.tag_configure("alta", background="#d4edda")
        coef_tree.tag_configure("media", background="#fff3cd")
        coef_tree.tag_configure("baja", background="#f8d7da")
        
        # Añadir barra de desplazamiento
        scrollbar = ttk.Scrollbar(coef_frame_inner, orient="vertical", command=coef_tree.yview)
        coef_tree.configure(yscrollcommand=scrollbar.set)
        
        # Colocar tabla y scrollbar
        scrollbar.pack(side="right", fill="y")
        coef_tree.pack(side="left", fill="both", expand=True)
        
        # Crear figura para el gráfico de coeficientes
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Graficar coeficientes
        coef_values = params
        var_names_plot = var_names
        
        # Ordenar por magnitud absoluta
        sorted_indices = np.argsort(np.abs(coef_values))[::-1]
        sorted_coefs = coef_values[sorted_indices]
        sorted_vars = [var_names_plot[i] for i in sorted_indices]
        
        # Graficar barras
        bars = ax.barh(range(len(sorted_vars)), sorted_coefs, align='center', alpha=0.7)
        
        # Colorear barras según signo
        for i, bar in enumerate(bars):
            if sorted_coefs[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax.set_yticks(range(len(sorted_vars)))
        ax.set_yticklabels(sorted_vars)
        ax.set_xlabel('Coeficiente')
        ax.set_title('Coeficientes del Modelo GLS')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.coef_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_diagnostics(self):
        """Muestra los gráficos de diagnóstico del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.diagnostics_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.diagnostics_tests_frame.winfo_children():
            widget.destroy()
        
        try:
            # Crear figura para diagnósticos
            fig = Figure(figsize=(10, 12))
            
            # Predicciones vs valores reales (entrenamiento)
            ax1 = fig.add_subplot(321)
            ax1.scatter(self.y_train, self.y_train_pred, color='blue', alpha=0.7)
            ax1.plot([min(self.y_train), max(self.y_train)], [min(self.y_train), max(self.y_train)], 'r--')
            ax1.set_xlabel('Valores Reales (Entrenamiento)')
            ax1.set_ylabel('Valores Predichos')
            ax1.set_title('Predicciones vs Valores Reales (Entrenamiento)')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Predicciones vs valores reales (prueba)
            ax2 = fig.add_subplot(322)
            ax2.scatter(self.y_test, self.y_test_pred, color='green', alpha=0.7)
            ax2.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
            ax2.set_xlabel('Valores Reales (Prueba)')
            ax2.set_ylabel('Valores Predichos')
            ax2.set_title('Predicciones vs Valores Reales (Prueba)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Residuos vs valores predichos (entrenamiento)
            ax3 = fig.add_subplot(323)
            ax3.scatter(self.y_train_pred, self.train_residuals, color='blue', alpha=0.7)
            ax3.axhline(y=0, color='red', linestyle='--')
            ax3.set_xlabel('Valores Predichos')
            ax3.set_ylabel('Residuos')
            ax3.set_title('Residuos vs Valores Predichos (Entrenamiento)')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Residuos vs valores predichos (prueba)
            ax4 = fig.add_subplot(324)
            ax4.scatter(self.y_test_pred, self.test_residuals, color='green', alpha=0.7)
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.set_xlabel('Valores Predichos')
            ax4.set_ylabel('Residuos')
            ax4.set_title('Residuos vs Valores Predichos (Prueba)')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Autocorrelación de residuos
            ax5 = fig.add_subplot(325)
            plot_acf(self.train_residuals, ax=ax5, lags=min(20, len(self.train_residuals) // 2), alpha=0.05)
            ax5.set_title('Autocorrelación de Residuos')
            
            # Autocorrelación parcial de residuos
            ax6 = fig.add_subplot(326)
            plot_pacf(self.train_residuals, ax=ax6, lags=min(20, len(self.train_residuals) // 2), alpha=0.05)
            ax6.set_title('Autocorrelación Parcial de Residuos')
            
            fig.tight_layout()
            
            # Incrustar el gráfico en la ventana tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.diagnostics_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Crear marco para pruebas de diagnóstico
            tests_frame = tk.Frame(self.diagnostics_tests_frame, bg="#f5f8fa")
            tests_frame.pack(fill=tk.BOTH, expand=True)
            
            # Prueba de Breusch-Pagan para heteroscedasticidad
            try:
                bp_test = het_breuschpagan(self.train_residuals, self.X_train)
                bp_pvalue = bp_test[1]
                
                bp_result = "Heteroscedasticidad detectada" if bp_pvalue < 0.05 else "No se detecta heteroscedasticidad"
                bp_text = f"Prueba de Breusch-Pagan para heteroscedasticidad:\nEstadístico: {bp_test[0]:.4f}, p-valor: {bp_pvalue:.4f}\nResultado: {bp_result}"
            except:
                bp_text = "No se pudo realizar la prueba de Breusch-Pagan"
            
            # Prueba de Ljung-Box para autocorrelación
            try:
                lb_test = acorr_ljungbox(self.train_residuals, lags=[10])
                lb_pvalue = lb_test[1][0]
                
                lb_result = "Autocorrelación detectada" if lb_pvalue < 0.05 else "No se detecta autocorrelación"
                lb_text = f"Prueba de Ljung-Box para autocorrelación:\nEstadístico: {lb_test[0][0]:.4f}, p-valor: {lb_pvalue:.4f}\nResultado: {lb_result}"
            except:
                lb_text = "No se pudo realizar la prueba de Ljung-Box"
            
            # Mostrar resultados de las pruebas
            tests_text = tk.Text(tests_frame, height=10, width=80, font=("Consolas", 10), 
                               bg="#ffffff", fg="#333333", wrap=tk.WORD)
            tests_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
            
            tests_text.insert(tk.END, "PRUEBAS DE DIAGNÓSTICO DEL MODELO\n")
            tests_text.insert(tk.END, "=" * 40 + "\n\n")
            tests_text.insert(tk.END, bp_text + "\n\n")
            tests_text.insert(tk.END, lb_text + "\n\n")
            
            # Interpretación de las pruebas
            tests_text.insert(tk.END, "INTERPRETACIÓN DE LAS PRUEBAS\n")
            tests_text.insert(tk.END, "=" * 40 + "\n\n")
            
            if bp_pvalue < 0.05:
                tests_text.insert(tk.END, "La prueba de Breusch-Pagan indica presencia de heteroscedasticidad. ")
                tests_text.insert(tk.END, "Esto sugiere que la varianza de los errores no es constante, ")
                tests_text.insert(tk.END, "lo que justifica el uso del modelo GLS en lugar de OLS.\n\n")
            else:
                tests_text.insert(tk.END, "La prueba de Breusch-Pagan no detecta heteroscedasticidad significativa. ")
                tests_text.insert(tk.END, "La varianza de los errores parece ser constante.\n\n")
            
            if lb_pvalue < 0.05:
                tests_text.insert(tk.END, "La prueba de Ljung-Box indica presencia de autocorrelación. ")
                tests_text.insert(tk.END, "Esto sugiere que los errores están correlacionados en el tiempo, ")
                if self.auto_params_var.get():
                    tests_text.insert(tk.END, f"lo que justifica el uso de la estructura de correlación {self.best_corr_structure}.\n")
                else:
                    tests_text.insert(tk.END, f"lo que justifica el uso de la estructura de correlación {self.corr_structure_var.get()}.\n")
            else:
                tests_text.insert(tk.END, "La prueba de Ljung-Box no detecta autocorrelación significativa. ")
                tests_text.insert(tk.END, "Los errores parecen ser independientes en el tiempo.\n")
            
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
        self.display_coefficients()
        self.display_diagnostics()
        
        # Cambiar a la pestaña de resumen
        self.notebook.select(self.summary_tab)
        
        # Mostrar resumen de predicción
        target_column = self.target_var.get()
        last_day = future_data["Tiempo_dias"].max()
        last_length = future_data[target_column].iloc[-1]
        messagebox.showinfo("Predicción Completada", 
                           f"¡Predicción completada con éxito!\n\n"
                           f"En el día {last_day}, la longitud predicha del pez es {last_length:.4f} cm.\n\n"
                           f"Crecimiento total en {days_to_predict} días: {last_length - self.data[target_column].iloc[-1]:.4f} cm")
        
        self.status_var.set(f"Predicción completada. Longitud final: {last_length:.4f} cm")

if __name__ == "__main__":
    root = tk.Tk()
    app = GLSFishPredictor(root)
    root.mainloop()