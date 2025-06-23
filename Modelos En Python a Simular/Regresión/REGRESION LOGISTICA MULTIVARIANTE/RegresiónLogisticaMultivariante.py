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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

class LogisticFishPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Regresión Logística")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.X = None  # Variables predictoras
        self.y = None  # Variable objetivo (categoría de crecimiento)
        self.y_numeric = None  # Variable objetivo numérica (longitud)
        self.model = None
        self.scaler_X = StandardScaler()
        self.LIMITE_BIOLOGICO = 70  # Límite biológico en cm para truchas
        
        # Categorías de crecimiento
        self.growth_categories = ['Lento', 'Normal', 'Rápido']
        
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
        
        tk.Label(title_frame, text="Modelo de Regresión Logística Multivariante", 
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
        
        # Columna 2: Parámetros del modelo
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Parámetros del modelo:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Método de validación cruzada
        cv_frame = tk.Frame(col2, bg="#f5f8fa")
        cv_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(cv_frame, text="Validación cruzada (k-fold):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.cv_folds_var = tk.StringVar(value="5")
        tk.Entry(cv_frame, textvariable=self.cv_folds_var, width=8).pack(side="left")
        
        # Tamaño del conjunto de prueba
        test_frame = tk.Frame(col2, bg="#f5f8fa")
        test_frame.pack(anchor="w")
        
        tk.Label(test_frame, text="Tamaño del conjunto de prueba (%):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.test_size_var = tk.StringVar(value="20")
        tk.Entry(test_frame, textvariable=self.test_size_var, width=8).pack(side="left")
        
        # Umbral de crecimiento
        threshold_frame = tk.Frame(col2, bg="#f5f8fa")
        threshold_frame.pack(anchor="w", pady=(5, 0))
        
        tk.Label(threshold_frame, text="Umbral crecimiento rápido (cm/día):", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.rapid_threshold_var = tk.StringVar(value="0.15")
        tk.Entry(threshold_frame, textvariable=self.rapid_threshold_var, width=8).pack(side="left")
        
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
        
        # Pestaña para importancia de variables
        self.variables_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.variables_tab, text="Importancia de Variables")
        
        # Pestaña para diagnóstico del modelo
        self.diagnostics_tab = tk.Frame(self.notebook, bg="#f5f8fa")
        self.notebook.add(self.diagnostics_tab, text="Diagnóstico del Modelo")
        
        # Configurar contenido de las pestañas
        self.setup_summary_tab()
        self.setup_predictions_tab()
        self.setup_growth_tab()
        self.setup_variables_tab()
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
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo de Regresión Logística", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """La Regresión Logística Multivariante es una técnica estadística que permite clasificar observaciones en diferentes categorías basándose en múltiples variables predictoras. A diferencia de la regresión lineal, que predice valores continuos, la regresión logística predice probabilidades de pertenencia a categorías.

En este modelo:

1. Se clasifican los patrones de crecimiento en tres categorías: Lento, Normal y Rápido
2. El modelo calcula la probabilidad de que un pez pertenezca a cada categoría de crecimiento
3. Se utilizan estas probabilidades para estimar el crecimiento futuro y la longitud esperada
4. Las variables predictoras (como tiempo, condiciones ambientales, alimentación) se ponderan según su importancia

La regresión logística es especialmente útil cuando se necesita entender qué factores influyen en la clasificación y cuantificar su impacto, proporcionando resultados fácilmente interpretables."""
        
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
        columns = ("Día", "Longitud (cm)", "Crecimiento (cm)", "Tasa (%)", "Categoría", "Probabilidad (%)")
        self.predictions_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        # Configurar encabezados y anchos de columna
        self.predictions_tree.heading("Día", text="Día")
        self.predictions_tree.column("Día", width=60, anchor="center")
        
        self.predictions_tree.heading("Longitud (cm)", text="Longitud (cm)")
        self.predictions_tree.column("Longitud (cm)", width=100, anchor="center")
        
        self.predictions_tree.heading("Crecimiento (cm)", text="Crecimiento (cm)")
        self.predictions_tree.column("Crecimiento (cm)", width=100, anchor="center")
        
        self.predictions_tree.heading("Tasa (%)", text="Tasa (%)")
        self.predictions_tree.column("Tasa (%)", width=80, anchor="center")
        
        self.predictions_tree.heading("Categoría", text="Categoría")
        self.predictions_tree.column("Categoría", width=100, anchor="center")
        
        self.predictions_tree.heading("Probabilidad (%)", text="Probabilidad (%)")
        self.predictions_tree.column("Probabilidad (%)", width=120, anchor="center")
        
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
        
        # Marco para el gráfico de probabilidades
        self.prob_plot_frame = tk.LabelFrame(growth_frame, text="Probabilidades de Categorías de Crecimiento", 
                                          font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                          padx=10, pady=10)
        self.prob_plot_frame.pack(fill="both", expand=True, pady=(15, 0))
    
    def setup_variables_tab(self):
        """Configura la pestaña de importancia de variables"""
        variables_frame = tk.Frame(self.variables_tab, bg="#f5f8fa", padx=15, pady=15)
        variables_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de importancia de variables
        self.variables_plot_frame = tk.LabelFrame(variables_frame, text="Importancia de Variables", 
                                               font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                               padx=10, pady=10)
        self.variables_plot_frame.pack(fill="both", expand=True)
        
        # Marco para la tabla de coeficientes
        self.coef_frame = tk.LabelFrame(variables_frame, text="Coeficientes del Modelo", 
                                      font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                      padx=10, pady=10)
        self.coef_frame.pack(fill="both", expand=True, pady=(15, 0))
    
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
            
            # Calcular tasas de crecimiento para categorización
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
            return False
    
    def prepare_data(self):
        """Prepara los datos para el modelo de regresión logística"""
        try:
            target_column = self.target_var.get()
            
            # Guardar la variable objetivo numérica
            self.y_numeric = self.data[target_column].values
            
            # Crear categorías de crecimiento basadas en la tasa de crecimiento
            try:
                rapid_threshold = float(self.rapid_threshold_var.get())
            except:
                rapid_threshold = 0.15  # Valor por defecto
            
            slow_threshold = rapid_threshold / 3
            
            # Crear columna de categoría de crecimiento
            self.data["Growth_Category"] = "Normal"
            self.data.loc[self.data["Growth_Rate"] < slow_threshold, "Growth_Category"] = "Lento"
            self.data.loc[self.data["Growth_Rate"] > rapid_threshold, "Growth_Category"] = "Rápido"
            
            # Separar variables predictoras y objetivo
            self.y = self.data["Growth_Category"].values
            
            # Excluir la variable objetivo y otras columnas derivadas
            exclude_cols = [target_column, "Growth_Category", "Growth_Rate", "Growth_Rate_Pct"]
            self.X = self.data.drop(columns=exclude_cols).values
            
            # Guardar nombres de columnas para interpretación
            self.feature_names = self.data.drop(columns=exclude_cols).columns.tolist()
            
            # Normalizar datos si está activado
            if self.normalize_var.get():
                self.X = self.scaler_X.fit_transform(self.X)
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar datos: {str(e)}")
            return False
    
    def train_model(self):
        """Entrena el modelo de regresión logística"""
        try:
            self.status_var.set("Entrenando modelo de regresión logística...")
            self.root.update()
            
            # Dividir datos en entrenamiento y prueba
            test_size = float(self.test_size_var.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42)
            
            # Entrenar modelo de regresión logística
            self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            self.model.fit(X_train, y_train)
            
            # Guardar datos de entrenamiento y prueba
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            
            # Calcular predicciones en datos de entrenamiento y prueba
            self.y_train_pred = self.model.predict(X_train)
            self.y_test_pred = self.model.predict(X_test)
            
            # Calcular probabilidades
            self.y_train_proba = self.model.predict_proba(X_train)
            self.y_test_proba = self.model.predict_proba(X_test)
            
            # Calcular métricas de clasificación
            self.train_accuracy = accuracy_score(y_train, self.y_train_pred)
            self.test_accuracy = accuracy_score(y_test, self.y_test_pred)
            
            # Calcular matriz de confusión
            self.train_cm = confusion_matrix(y_train, self.y_train_pred)
            self.test_cm = confusion_matrix(y_test, self.y_test_pred)
            
            # Calcular reporte de clasificación
            self.train_report = classification_report(y_train, self.y_train_pred, output_dict=True)
            self.test_report = classification_report(y_test, self.y_test_pred, output_dict=True)
            
            self.status_var.set("Modelo de regresión logística entrenado con éxito")
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
            categories = []
            probabilities = []
            growth_rates = []
            
            # Realizar predicciones iterativas
            for i in range(days_to_predict):
                # Crear un registro para predecir
                X_pred = np.zeros((1, len(self.feature_names)))
                
                # Llenar con los valores del último registro conocido
                for j, feature in enumerate(self.feature_names):
                    if feature == "Tiempo_dias":
                        X_pred[0, j] = future_days[i]
                    else:
                        X_pred[0, j] = last_record[feature]
                
                # Normalizar si es necesario
                if self.normalize_var.get():
                    X_pred = self.scaler_X.transform(X_pred)
                
                # Predecir categoría de crecimiento
                category = self.model.predict(X_pred)[0]
                
                # Predecir probabilidades
                proba = self.model.predict_proba(X_pred)[0]
                max_proba = np.max(proba) * 100  # Convertir a porcentaje
                
                # Estimar tasa de crecimiento basada en la categoría
                if category == "Lento":
                    growth_rate = 0.05  # Crecimiento lento
                elif category == "Rápido":
                    growth_rate = 0.2   # Crecimiento rápido
                else:
                    growth_rate = 0.1   # Crecimiento normal
                
                # Ajustar la tasa de crecimiento basada en la probabilidad
                category_idx = self.growth_categories.index(category)
                growth_rate = growth_rate * (proba[category_idx] + 0.5)  # Ajustar según confianza
                
                # Calcular nueva longitud
                if i == 0:
                    prev_length = last_record[target_column]
                else:
                    prev_length = predictions[i-1]
                
                new_length = prev_length + growth_rate
                
                # Aplicar restricciones biológicas si está activado
                if self.bio_constraints_var.get():
                    # Limitar al límite biológico
                    new_length = min(new_length, self.LIMITE_BIOLOGICO)
                    
                    # Desacelerar crecimiento cerca del límite biológico
                    if new_length > self.LIMITE_BIOLOGICO * 0.95:
                        distancia_al_limite = self.LIMITE_BIOLOGICO - prev_length
                        crecimiento = distancia_al_limite * 0.05  # 5% de la distancia restante al límite
                        new_length = prev_length + crecimiento
                        growth_rate = crecimiento
                
                # Guardar predicciones
                predictions.append(new_length)
                categories.append(category)
                probabilities.append(max_proba)
                growth_rates.append(growth_rate)
                
                # Actualizar el último registro con la predicción
                last_record[target_column] = new_length
            
            # Añadir predicciones al dataframe
            future_data[target_column] = predictions
            future_data["Growth_Category"] = categories
            future_data["Probability"] = probabilities
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
        metrics_str += f"Exactitud (Accuracy): {self.train_accuracy:.4f}\n\n"
        
        metrics_str += "Métricas por categoría:\n"
        for category in self.growth_categories:
            if category in self.train_report:
                metrics_str += f"  {category}:\n"
                metrics_str += f"    Precisión: {self.train_report[category]['precision']:.4f}\n"
                metrics_str += f"    Sensibilidad: {self.train_report[category]['recall']:.4f}\n"
                metrics_str += f"    F1-Score: {self.train_report[category]['f1-score']:.4f}\n\n"
        
        metrics_str += "Conjunto de Prueba:\n"
        metrics_str += f"Exactitud (Accuracy): {self.test_accuracy:.4f}\n\n"
        
        metrics_str += "Métricas por categoría:\n"
        for category in self.growth_categories:
            if category in self.test_report:
                metrics_str += f"  {category}:\n"
                metrics_str += f"    Precisión: {self.test_report[category]['precision']:.4f}\n"
                metrics_str += f"    Sensibilidad: {self.test_report[category]['recall']:.4f}\n"
                metrics_str += f"    F1-Score: {self.test_report[category]['f1-score']:.4f}\n\n"
        
        metrics_str += "INTERPRETACIÓN DE MÉTRICAS\n"
        metrics_str += "=" * 40 + "\n\n"
        
        # Interpretación de exactitud
        if self.test_accuracy >= 0.9:
            metrics_str += "Exactitud: Excelente capacidad de clasificación.\n"
        elif self.test_accuracy >= 0.8:
            metrics_str += "Exactitud: Muy buena capacidad de clasificación.\n"
        elif self.test_accuracy >= 0.7:
            metrics_str += "Exactitud: Buena capacidad de clasificación.\n"
        elif self.test_accuracy >= 0.6:
            metrics_str += "Exactitud: Capacidad de clasificación moderada.\n"
        else:
            metrics_str += "Exactitud: Capacidad de clasificación limitada.\n"
        
        # Comparación entre entrenamiento y prueba
        if self.train_accuracy - self.test_accuracy > 0.2:
            metrics_str += "\nAdvertencia: La diferencia entre exactitud de entrenamiento y prueba sugiere sobreajuste.\n"
        
        self.metrics_text.insert(tk.END, metrics_str)
    
    def display_parameters(self):
        """Muestra los parámetros del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.params_text.delete(1.0, tk.END)
        
        # Formatear parámetros
        params_str = "PARÁMETROS DEL MODELO DE REGRESIÓN LOGÍSTICA\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"Número de variables predictoras: {len(self.feature_names)}\n"
        params_str += f"Tamaño del conjunto de entrenamiento: {len(self.y_train)} registros\n"
        params_str += f"Tamaño del conjunto de prueba: {len(self.y_test)} registros\n\n"
        
        params_str += "Variables predictoras utilizadas:\n"
        for i, feature in enumerate(self.feature_names):
            params_str += f"- {feature}\n"
        
        params_str += "\nCategorías de crecimiento:\n"
        for category in self.growth_categories:
            count = np.sum(self.y == category)
            percentage = count / len(self.y) * 100
            params_str += f"- {category}: {count} registros ({percentage:.1f}%)\n"
        
        params_str += "\nMatriz de confusión (Prueba):\n"
        params_str += "                  Predicción\n"
        params_str += "                  " + " ".join([f"{cat:<10}" for cat in self.growth_categories]) + "\n"
        params_str += "Real\n"
        
        for i, category in enumerate(self.growth_categories):
            params_str += f"{category:<10}      "
            for j in range(len(self.growth_categories)):
                if i < self.test_cm.shape[0] and j < self.test_cm.shape[1]:
                    params_str += f"{self.test_cm[i, j]:<10}"
                else:
                    params_str += f"0         "
            params_str += "\n"
        
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
            category = row["Growth_Category"]
            probability = row["Probability"]
            
            # Determinar tag para colorear
            if category == "Rápido":
                tag = "rapido"
            elif category == "Lento":
                tag = "lento"
            else:
                tag = "normal"
            
            # Insertar fila en la tabla
            item_id = self.predictions_tree.insert("", "end", values=(
                day,
                f"{length:.2f}",
                f"{growth:.4f}",
                f"{growth_pct:.2f}",
                category,
                f"{probability:.1f}"
            ))
            
            # Aplicar etiqueta para colorear
            self.predictions_tree.item(item_id, tags=(tag,))
        
        # Configurar colores para los tags
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
        stats_str += f"Crecimiento total en {len(future_data)} días: {future_data[target_column].iloc[-1] - last_known_length:.2f} cm\n\n"
        
        # Distribución de categorías
        stats_str += "Distribución de categorías de crecimiento:\n"
        category_counts = future_data["Growth_Category"].value_counts()
        for category, count in category_counts.items():
            percentage = count / len(future_data) * 100
            stats_str += f"- {category}: {count} días ({percentage:.1f}%)\n"
        
        self.growth_stats_text.insert(tk.END, stats_str)
    
    def display_growth_plot(self, future_data):
        """Muestra el gráfico de crecimiento"""
        # Limpiar gráficos anteriores
        for widget in self.growth_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.prob_plot_frame.winfo_children():
            widget.destroy()
        
        target_column = self.target_var.get()
        
        # Crear figura para el gráfico de crecimiento
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Graficar datos originales
        ax1.scatter(self.data["Tiempo_dias"], self.data[target_column], 
                  color='blue', label='Datos Originales', s=30, alpha=0.7)
        
        # Graficar datos futuros predichos con colores por categoría
        colors = {'Lento': 'orange', 'Normal': 'green', 'Rápido': 'red'}
        
        for category in self.growth_categories:
            mask = future_data["Growth_Category"] == category
            if mask.any():
                ax1.scatter(future_data.loc[mask, "Tiempo_dias"], 
                          future_data.loc[mask, target_column],
                          color=colors[category], 
                          label=f'Predicción ({category})', 
                          s=30, alpha=0.7)
        
        # Conectar los puntos con líneas
        all_days = list(self.data["Tiempo_dias"]) + list(future_data["Tiempo_dias"])
        all_lengths = list(self.data[target_column]) + list(future_data[target_column])
        ax1.plot(all_days, all_lengths, 'k--', alpha=0.5)
        
        # Añadir línea de límite biológico
        ax1.axhline(y=self.LIMITE_BIOLOGICO, color='red', linestyle='--', alpha=0.5, 
                   label=f'Límite Biológico ({self.LIMITE_BIOLOGICO} cm)')
        
        ax1.set_xlabel('Tiempo (días)')
        ax1.set_ylabel(f'{target_column}')
        ax1.set_title('Predicción de Crecimiento - Modelo de Regresión Logística')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.growth_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear figura para el gráfico de probabilidades
        fig2 = Figure(figsize=(10, 4))
        ax2 = fig2.add_subplot(111)
        
        # Graficar probabilidades por categoría
        bottom = np.zeros(len(future_data))
        
        # Obtener probabilidades para cada categoría
        probs_by_category = {}
        for i, category in enumerate(self.growth_categories):
            # Asignar probabilidades basadas en la categoría predicha
            probs = np.zeros(len(future_data))
            for j, row_category in enumerate(future_data["Growth_Category"]):
                if row_category == category:
                    probs[j] = future_data["Probability"].iloc[j] / 100  # Convertir de porcentaje a proporción
                else:
                    # Asignar probabilidades menores a las otras categorías
                    if row_category == "Normal" and category in ["Lento", "Rápido"]:
                        probs[j] = 0.2
                    elif row_category in ["Lento", "Rápido"] and category == "Normal":
                        probs[j] = 0.3
                    else:
                        probs[j] = 0.1
            
            probs_by_category[category] = probs
        
        # Graficar barras apiladas
        for category in self.growth_categories:
            ax2.bar(future_data["Tiempo_dias"], probs_by_category[category], 
                   bottom=bottom, label=category, alpha=0.7, color=colors[category])
            bottom += probs_by_category[category]
        
        ax2.set_xlabel('Tiempo (días)')
        ax2.set_ylabel('Probabilidad')
        ax2.set_title('Probabilidades de Categorías de Crecimiento')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Incrustar el gráfico en la ventana tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=self.prob_plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_variables_importance(self):
        """Muestra la importancia de las variables"""
        # Limpiar gráficos anteriores
        for widget in self.variables_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.coef_frame.winfo_children():
            widget.destroy()
        
        # Crear figura para el gráfico de importancia de variables
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Obtener coeficientes del modelo
        coefs = np.abs(self.model.coef_).mean(axis=0)  # Promedio de los coeficientes para todas las clases
        
        # Ordenar variables por importancia
        indices = np.argsort(coefs)[::-1]
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_coefs = coefs[indices]
        
        # Graficar importancia de variables
        bars = ax.barh(range(len(sorted_features)), sorted_coefs, align='center', alpha=0.7)
        
        # Colorear barras según importancia
        for i, bar in enumerate(bars):
            if i < len(sorted_features) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_features) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importancia (Magnitud de Coeficientes)')
        ax.set_title('Importancia de Variables en el Modelo de Regresión Logística')
        
        # Ajustar márgenes
        fig.tight_layout()
        
        # Incrustar el gráfico en la ventana tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.variables_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear tabla de coeficientes
        coef_frame_inner = tk.Frame(self.coef_frame, bg="#f5f8fa")
        coef_frame_inner.pack(fill=tk.BOTH, expand=True)
        
        # Crear tabla con Treeview
        columns = ["Variable"] + self.growth_categories + ["Importancia"]
        coef_tree = ttk.Treeview(coef_frame_inner, columns=columns, show="headings", height=15)
        
        # Configurar encabezados
        for col in columns:
            coef_tree.heading(col, text=col)
            if col == "Variable":
                coef_tree.column(col, width=150, anchor="w")
            else:
                coef_tree.column(col, width=100, anchor="center")
        
        # Añadir datos a la tabla
        for i, feature in enumerate(self.feature_names):
            # Determinar importancia
            importance = coefs[i]
            if i < len(self.feature_names) // 3:
                importancia = "Alta"
                tag = "alta"
            elif i < 2 * len(self.feature_names) // 3:
                importancia = "Media"
                tag = "media"
            else:
                importancia = "Baja"
                tag = "baja"
            
            # Obtener coeficientes para cada categoría
            category_coefs = []
            for j, category in enumerate(self.growth_categories):
                if j < self.model.coef_.shape[0]:
                    category_coefs.append(f"{self.model.coef_[j, i]:.4f}")
                else:
                    category_coefs.append("N/A")
            
            # Insertar fila en la tabla
            values = [feature] + category_coefs + [importancia]
            item_id = coef_tree.insert("", "end", values=values)
            
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
    
    def display_diagnostics_plot(self):
        """Muestra los gráficos de diagnóstico del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.diagnostics_plot_frame.winfo_children():
            widget.destroy()
        
        try:
            # Crear figura para diagnósticos
            fig = Figure(figsize=(10, 12))
            
            # Matriz de confusión (entrenamiento)
            ax1 = fig.add_subplot(221)
            im1 = ax1.imshow(self.train_cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax1.set_title('Matriz de Confusión (Entrenamiento)')
            
            # Añadir etiquetas
            tick_marks = np.arange(len(self.growth_categories))
            ax1.set_xticks(tick_marks)
            ax1.set_yticks(tick_marks)
            ax1.set_xticklabels(self.growth_categories)
            ax1.set_yticklabels(self.growth_categories)
            
            # Añadir valores a la matriz
            for i in range(self.train_cm.shape[0]):
                for j in range(self.train_cm.shape[1]):
                    ax1.text(j, i, str(self.train_cm[i, j]), 
                           ha="center", va="center", 
                           color="white" if self.train_cm[i, j] > self.train_cm.max() / 2 else "black")
            
            ax1.set_xlabel('Predicción')
            ax1.set_ylabel('Real')
            
            # Matriz de confusión (prueba)
            ax2 = fig.add_subplot(222)
            im2 = ax2.imshow(self.test_cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.set_title('Matriz de Confusión (Prueba)')
            
            # Añadir etiquetas
            ax2.set_xticks(tick_marks)
            ax2.set_yticks(tick_marks)
            ax2.set_xticklabels(self.growth_categories)
            ax2.set_yticklabels(self.growth_categories)
            
            # Añadir valores a la matriz
            for i in range(self.test_cm.shape[0]):
                for j in range(self.test_cm.shape[1]):
                    ax2.text(j, i, str(self.test_cm[i, j]), 
                           ha="center", va="center", 
                           color="white" if self.test_cm[i, j] > self.test_cm.max() / 2 else "black")
            
            ax2.set_xlabel('Predicción')
            ax2.set_ylabel('Real')
            
            # Distribución de probabilidades (entrenamiento)
            ax3 = fig.add_subplot(223)
            
            for i, category in enumerate(self.growth_categories):
                if i < self.y_train_proba.shape[1]:
                    # Filtrar probabilidades para la categoría actual
                    mask = self.y_train == category
                    if np.any(mask):
                        probs = self.y_train_proba[mask, i]
                        ax3.hist(probs, bins=10, alpha=0.5, label=category)
            
            ax3.set_title('Distribución de Probabilidades (Entrenamiento)')
            ax3.set_xlabel('Probabilidad')
            ax3.set_ylabel('Frecuencia')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Distribución de probabilidades (prueba)
            ax4 = fig.add_subplot(224)
            
            for i, category in enumerate(self.growth_categories):
                if i < self.y_test_proba.shape[1]:
                    # Filtrar probabilidades para la categoría actual
                    mask = self.y_test == category
                    if np.any(mask):
                        probs = self.y_test_proba[mask, i]
                        ax4.hist(probs, bins=10, alpha=0.5, label=category)
            
            ax4.set_title('Distribución de Probabilidades (Prueba)')
            ax4.set_xlabel('Probabilidad')
            ax4.set_ylabel('Frecuencia')
            ax4.legend()
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
        self.display_variables_importance()
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
                           f"Crecimiento total en {days_to_predict} días: {last_length - self.data[target_column].iloc[-1]:.4f} cm")
        
        self.status_var.set(f"Predicción completada. Longitud final: {last_length:.4f} cm")

if __name__ == "__main__":
    root = tk.Tk()
    app = LogisticFishPredictor(root)
    root.mainloop()