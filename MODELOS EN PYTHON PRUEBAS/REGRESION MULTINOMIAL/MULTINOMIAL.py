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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

class MultinomialFishPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Regresión Multinomial")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.X = None  # Variables predictoras
        self.y = None  # Variable objetivo (categoría de crecimiento)
        self.model = None
        self.scaler_X = StandardScaler()
        self.label_encoder = LabelEncoder()
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
        
        tk.Label(title_frame, text="Modelo de Regresión Multinomial", 
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
        
        # Columna 2: Parámetros del modelo Multinomial
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Parámetros del modelo Multinomial:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Número de categorías
        cat_frame = tk.Frame(col2, bg="#f5f8fa")
        cat_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(cat_frame, text="Número de categorías:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.n_categories_var = tk.StringVar(value="3")
        tk.Entry(cat_frame, textvariable=self.n_categories_var, width=8).pack(side="left")
        
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
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo de Regresión Multinomial", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """La Regresión Logística Multinomial es una técnica de clasificación que extiende la regresión logística binaria a problemas con múltiples clases. Es especialmente útil cuando:

1. La variable objetivo es categórica con más de dos categorías (por ejemplo, categorías de crecimiento: "bajo", "normal", "alto")
2. Se necesita predecir la probabilidad de pertenencia a cada categoría

El modelo funciona calculando la probabilidad de que una observación pertenezca a cada una de las categorías posibles, basándose en las variables predictoras. Características principales:

- Utiliza la función softmax para calcular probabilidades para múltiples categorías
- Permite interpretar la importancia relativa de cada variable predictora para cada categoría
- Es eficiente computacionalmente y fácil de interpretar
- Proporciona probabilidades de pertenencia a cada categoría, lo que permite una interpretación más rica de los resultados

A diferencia de la regresión PLS, que predice valores continuos, la regresión multinomial predice categorías discretas, lo que puede ser más apropiado para clasificar el crecimiento de los peces en categorías cualitativas."""
        
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
        columns = ("Día", "Categoría Predicha", "Probabilidad", "Longitud Estimada (cm)", "Estado")
        self.predictions_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        # Configurar encabezados y anchos de columna
        self.predictions_tree.heading("Día", text="Día")
        self.predictions_tree.column("Día", width=80, anchor="center")
        
        self.predictions_tree.heading("Categoría Predicha", text="Categoría Predicha")
        self.predictions_tree.column("Categoría Predicha", width=150, anchor="center")
        
        self.predictions_tree.heading("Probabilidad", text="Probabilidad")
        self.predictions_tree.column("Probabilidad", width=100, anchor="center")
        
        self.predictions_tree.heading("Longitud Estimada (cm)", text="Longitud Estimada (cm)")
        self.predictions_tree.column("Longitud Estimada (cm)", width=150, anchor="center")
        
        self.predictions_tree.heading("Estado", text="Estado")
        self.predictions_tree.column("Estado", width=150, anchor="center")
        
        # Añadir barra de desplazamiento
        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.predictions_tree.yview)
        self.predictions_tree.configure(yscrollcommand=table_scroll.set)
        
        # Colocar tabla y scrollbar
        table_scroll.pack(side="right", fill="y")
        self.predictions_tree.pack(side="left", fill="both", expand=True)
        
        # Marco para estadísticas de crecimiento
        stats_frame = tk.LabelFrame(predictions_frame, text="Estadísticas de Predicción", font=("Arial", 11, "bold"), 
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
        self.growth_plot_frame = tk.LabelFrame(growth_frame, text="Distribución de Categorías de Crecimiento", 
                                            font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                            padx=10, pady=10)
        self.growth_plot_frame.pack(fill="both", expand=True)
        
        # Marco para el gráfico de probabilidades
        self.prob_plot_frame = tk.LabelFrame(growth_frame, text="Probabilidades de Categorías por Día", 
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
            
            # Crear categorías de crecimiento basadas en la variable objetivo
            n_categories = int(self.n_categories_var.get())
            
            # Crear categorías basadas en percentiles
            self.data['Growth_Category'] = pd.qcut(
                self.data[target_column], 
                q=n_categories, 
                labels=[f"Categoría {i+1}" for i in range(n_categories)]
            )
            
            # Guardar los límites de las categorías para uso posterior
            self.category_bounds = pd.qcut(self.data[target_column], q=n_categories, retbins=True)[1]
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros válidos")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            return False
    
    def prepare_data(self):
        """Prepara los datos para el modelo de regresión multinomial"""
        try:
            # Separar variables predictoras y objetivo
            self.y = self.data['Growth_Category'].values
            
            # Excluir la variable objetivo y otras columnas no relevantes
            target_column = self.target_var.get()
            exclude_cols = [target_column, 'Growth_Category']
            self.X = self.data.drop(columns=exclude_cols).values
            
            # Guardar nombres de columnas para interpretación
            self.feature_names = self.data.drop(columns=exclude_cols).columns.tolist()
            
            # Normalizar datos si está activado
            if self.normalize_var.get():
                self.X = self.scaler_X.fit_transform(self.X)
            
            # Codificar las categorías
            self.y = self.label_encoder.fit_transform(self.y)
            
            # Guardar las clases para interpretación
            self.classes_ = self.label_encoder.classes_
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar datos: {str(e)}")
            return False
    
    def train_model(self):
        """Entrena el modelo de regresión multinomial"""
        try:
            self.status_var.set("Entrenando modelo de regresión multinomial...")
            self.root.update()
            
            # Dividir datos en entrenamiento y prueba
            test_size = float(self.test_size_var.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42)
            
            # Entrenar modelo de regresión multinomial
            self.model = LogisticRegression(
                multi_class='multinomial',  # Usar regresión multinomial
                solver='lbfgs',            # Algoritmo de optimización
                max_iter=1000,
                C=1.0,                    # Parámetro de regularización
                random_state=42
            )
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
            
            # Calcular métricas en datos de entrenamiento
            self.train_accuracy = accuracy_score(y_train, self.y_train_pred)
            self.train_report = classification_report(y_train, self.y_train_pred, output_dict=True)
            self.train_cm = confusion_matrix(y_train, self.y_train_pred)
            
            # Calcular métricas en datos de prueba
            self.test_accuracy = accuracy_score(y_test, self.y_test_pred)
            self.test_report = classification_report(y_test, self.y_test_pred, output_dict=True)
            self.test_cm = confusion_matrix(y_test, self.y_test_pred)
            
            self.status_var.set("Modelo de regresión multinomial entrenado con éxito")
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
            probabilities = []
            estimated_lengths = []
            
            # Realizar predicciones iterativas
            for i in range(days_to_predict):
                # Crear un registro para predecir
                # IMPORTANTE: Solo usar las columnas que se usaron para entrenar el modelo
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
                
                # Realizar predicción
                y_pred = self.model.predict(X_pred)[0]
                y_proba = self.model.predict_proba(X_pred)[0]
                
                # Obtener la categoría predicha y su probabilidad
                category = self.label_encoder.inverse_transform([y_pred])[0]
                probability = y_proba[y_pred]
                
                # Estimar longitud basada en la categoría
                # Usamos el punto medio del rango de la categoría
                category_idx = list(self.classes_).index(category)
                if category_idx == 0:
                    # Primera categoría
                    estimated_length = (self.category_bounds[0] + self.category_bounds[1]) / 2
                elif category_idx == len(self.classes_) - 1:
                    # Última categoría
                    estimated_length = (self.category_bounds[-2] + self.category_bounds[-1]) / 2
                else:
                    # Categorías intermedias
                    estimated_length = (self.category_bounds[category_idx] + self.category_bounds[category_idx + 1]) / 2
                
                # Aplicar restricciones biológicas si está activado
                if self.bio_constraints_var.get():
                    # Asegurar que el pez no encoja
                    if i > 0:
                        estimated_length = max(estimated_length, estimated_lengths[i-1])
                    else:
                        estimated_length = max(estimated_length, last_record[target_column])
                    
                    # Limitar al límite biológico
                    estimated_length = min(estimated_length, self.LIMITE_BIOLOGICO)
                
                # Guardar predicción
                predictions.append(category)
                probabilities.append(probability)
                estimated_lengths.append(estimated_length)
                
                # Actualizar el último registro con la predicción
                last_record[target_column] = estimated_length
            
            # Añadir predicciones al dataframe
            future_data["Category"] = predictions
            future_data["Probability"] = probabilities
            future_data["Estimated_Length"] = estimated_lengths
            
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
        metrics_str += f"Exactitud (Accuracy): {self.train_accuracy:.4f}\n"
        metrics_str += f"   Interpretación: {self.train_accuracy * 100:.2f}% de predicciones correctas\n\n"
        
        metrics_str += "Métricas por categoría (Entrenamiento):\n"
        for category in self.classes_:
            category_idx = list(self.label_encoder.classes_).index(category)
            metrics_str += f"- {category}:\n"
            metrics_str += f"  Precisión: {self.train_report[str(category_idx)]['precision']:.4f}\n"
            metrics_str += f"  Sensibilidad: {self.train_report[str(category_idx)]['recall']:.4f}\n"
            metrics_str += f"  F1-Score: {self.train_report[str(category_idx)]['f1-score']:.4f}\n"
        
        metrics_str += "\nConjunto de Prueba:\n"
        metrics_str += f"Exactitud (Accuracy): {self.test_accuracy:.4f}\n"
        metrics_str += f"   Interpretación: {self.test_accuracy * 100:.2f}% de predicciones correctas\n\n"
        
        metrics_str += "Métricas por categoría (Prueba):\n"
        for category in self.classes_:
            category_idx = list(self.label_encoder.classes_).index(category)
            metrics_str += f"- {category}:\n"
            metrics_str += f"  Precisión: {self.test_report[str(category_idx)]['precision']:.4f}\n"
            metrics_str += f"  Sensibilidad: {self.test_report[str(category_idx)]['recall']:.4f}\n"
            metrics_str += f"  F1-Score: {self.test_report[str(category_idx)]['f1-score']:.4f}\n"
        
        metrics_str += "\nINTERPRETACIÓN DE MÉTRICAS\n"
        metrics_str += "=" * 40 + "\n\n"
        
        # Interpretación de Accuracy
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
        params_str = "PARÁMETROS DEL MODELO MULTINOMIAL\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"Número de categorías: {len(self.classes_)}\n"
        params_str += f"Categorías: {', '.join(self.classes_)}\n"
        params_str += f"Número de variables predictoras: {len(self.feature_names)}\n"
        params_str += f"Tamaño del conjunto de entrenamiento: {len(self.y_train)} registros\n"
        params_str += f"Tamaño del conjunto de prueba: {len(self.y_test)} registros\n\n"
        
        params_str += "Variables predictoras utilizadas:\n"
        for i, feature in enumerate(self.feature_names):
            params_str += f"- {feature}\n"
        
        params_str += "\nLímites de categorías (en términos de longitud):\n"
        for i in range(len(self.category_bounds) - 1):
            params_str += f"{self.classes_[i]}: {self.category_bounds[i]:.2f} - {self.category_bounds[i+1]:.2f} cm\n"
        
        params_str += "\nParámetros del modelo:\n"
        params_str += f"Solver: {self.model.solver}\n"
        params_str += f"Regularización (C): {self.model.C}\n"
        params_str += f"Iteraciones máximas: {self.model.max_iter}\n"
        
        self.params_text.insert(tk.END, params_str)
    
    def display_predictions_table(self, future_data):
        """Muestra la tabla de predicciones"""
        # Limpiar tabla anterior
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Limpiar estadísticas de crecimiento
        self.growth_stats_text.delete(1.0, tk.END)
        
        # Añadir datos a la tabla
        for i, row in future_data.iterrows():
            day = int(row["Tiempo_dias"])
            category = row["Category"]
            probability = row["Probability"]
            estimated_length = row["Estimated_Length"]
            
            # Determinar el estado del crecimiento
            if estimated_length >= self.LIMITE_BIOLOGICO * 0.95:
                estado = "Cerca del límite biológico"
                tag = "limite"
            elif category == self.classes_[-1]:  # Categoría más alta
                estado = "Crecimiento rápido"
                tag = "rapido"
            elif category == self.classes_[0]:  # Categoría más baja
                estado = "Crecimiento lento"
                tag = "lento"
            else:
                estado = "Crecimiento normal"
                tag = "normal"
            
            # Insertar fila en la tabla
            item_id = self.predictions_tree.insert("", "end", values=(
                day,
                category,
                f"{probability:.4f}",
                f"{estimated_length:.2f}",
                estado
            ))
            
            # Aplicar etiqueta para colorear
            self.predictions_tree.item(item_id, tags=(tag,))
        
        # Configurar colores para los tags
        self.predictions_tree.tag_configure("limite", background="#f8d7da")
        self.predictions_tree.tag_configure("rapido", background="#d4edda")
        self.predictions_tree.tag_configure("lento", background="#fff3cd")
        self.predictions_tree.tag_configure("normal", background="#e8f4f8")
        
        # Mostrar estadísticas de predicción
        category_counts = future_data["Category"].value_counts()
        
        stats_str = "ESTADÍSTICAS DE PREDICCIÓN\n"
        stats_str += "=" * 40 + "\n\n"
        
        stats_str += "Distribución de categorías predichas:\n"
        for category in self.classes_:
            count = category_counts.get(category, 0)
            percentage = (count / len(future_data)) * 100
            stats_str += f"{category}: {count} días ({percentage:.2f}%)\n"
        
        stats_str += f"\nLongitud estimada final: {future_data['Estimated_Length'].iloc[-1]:.2f} cm\n"
        stats_str += f"Categoría final predicha: {future_data['Category'].iloc[-1]}\n"
        stats_str += f"Confianza en la predicción final: {future_data['Probability'].iloc[-1]:.4f}\n"
        
        self.growth_stats_text.insert(tk.END, stats_str)
    
    def display_growth_plot(self, future_data):
        """Muestra el gráfico de crecimiento"""
        # Limpiar gráficos anteriores
        for widget in self.growth_plot_frame.winfo_children():
            widget.destroy()
        
        for widget in self.prob_plot_frame.winfo_children():
            widget.destroy()
        
        # Crear figura para el gráfico de distribución de categorías
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        # Contar categorías
        category_counts = future_data["Category"].value_counts().sort_index()
        
        # Graficar distribución de categorías
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.classes_)))
        bars = ax1.bar(category_counts.index, category_counts.values, color=colors)
        
        ax1.set_xlabel('Categoría de Crecimiento')
        ax1.set_ylabel('Número de Días')
        ax1.set_title('Distribución de Categorías de Crecimiento Predichas')
        
        # Añadir etiquetas con porcentajes
        for bar in bars:
            height = bar.get_height()
            percentage = (height / len(future_data)) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height} ({percentage:.1f}%)',
                   ha='center', va='bottom')
        
        # Incrustar el gráfico en la ventana tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.growth_plot_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Crear figura para el gráfico de probabilidades por día
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        # Graficar longitud estimada
        ax2.plot(future_data["Tiempo_dias"], future_data["Estimated_Length"], 
               'b-', linewidth=2, label='Longitud Estimada')
        
        # Añadir etiquetas de categoría en puntos clave
        for i in range(0, len(future_data), max(1, len(future_data) // 10)):
            ax2.annotate(
                future_data["Category"].iloc[i],
                (future_data["Tiempo_dias"].iloc[i], future_data["Estimated_Length"].iloc[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        
        # Añadir línea de límite biológico
        ax2.axhline(y=self.LIMITE_BIOLOGICO, color='red', linestyle='--', alpha=0.5, 
                   label=f'Límite Biológico ({self.LIMITE_BIOLOGICO} cm)')
        
        # Añadir líneas para los límites de categorías
        for i, bound in enumerate(self.category_bounds):
            if i > 0 and i < len(self.category_bounds) - 1:
                ax2.axhline(y=bound, color='green', linestyle=':', alpha=0.5)
                ax2.text(future_data["Tiempo_dias"].min(), bound, f'Límite {self.classes_[i-1]}/{self.classes_[i]}', 
                       va='bottom', ha='left', fontsize=8)
        
        ax2.set_xlabel('Tiempo (días)')
        ax2.set_ylabel('Longitud Estimada (cm)')
        ax2.set_title('Evolución de la Longitud Estimada y Categorías')
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
        coefs = self.model.coef_
        
        # Calcular importancia de variables (magnitud de los coeficientes)
        importance = np.abs(coefs).mean(axis=0)
        
        # Ordenar variables por importancia
        indices = np.argsort(importance)[::-1]
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        
        # Graficar importancia de variables
        bars = ax.barh(range(len(sorted_features)), sorted_importance, align='center', alpha=0.7)
        
        # Colorear barras según importancia
        norm = plt.Normalize(sorted_importance.min(), sorted_importance.max())
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(norm(sorted_importance[i])))
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importancia (Magnitud Media de Coeficientes)')
        ax.set_title('Importancia de Variables en el Modelo Multinomial')
        
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
        columns = ["Variable"] + [f"Coef. {cat}" for cat in self.classes_[1:]] + ["Importancia"]
        coef_tree = ttk.Treeview(coef_frame_inner, columns=columns, show="headings", height=15)
        
        # Configurar encabezados
        for col in columns:
            coef_tree.heading(col, text=col)
            coef_tree.column(col, width=120, anchor="center")
        
        # Añadir datos a la tabla
        for i, feature in enumerate(self.feature_names):
            # Obtener coeficientes para esta variable (uno por categoría)
            feature_coefs = coefs[:, i]
            
            # Calcular importancia
            feature_importance = importance[i]
            
            # Determinar nivel de importancia
            if feature_importance > np.percentile(importance, 75):
                importancia = "Alta"
                tag = "alta"
            elif feature_importance > np.percentile(importance, 50):
                importancia = "Media"
                tag = "media"
            else:
                importancia = "Baja"
                tag = "baja"
            
            # Preparar valores para la tabla
            values = [feature]
            for coef in feature_coefs:
                values.append(f"{coef:.6f}")
            values.append(importancia)
            
            # Insertar fila en la tabla
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
            tick_marks = np.arange(len(self.classes_))
            ax1.set_xticks(tick_marks)
            ax1.set_yticks(tick_marks)
            ax1.set_xticklabels(self.classes_, rotation=45, ha="right")
            ax1.set_yticklabels(self.classes_)
            
            # Añadir valores a las celdas
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
            ax2.set_xticklabels(self.classes_, rotation=45, ha="right")
            ax2.set_yticklabels(self.classes_)
            
            # Añadir valores a las celdas
            for i in range(self.test_cm.shape[0]):
                for j in range(self.test_cm.shape[1]):
                    ax2.text(j, i, str(self.test_cm[i, j]),
                           ha="center", va="center",
                           color="white" if self.test_cm[i, j] > self.test_cm.max() / 2 else "black")
            
            ax2.set_xlabel('Predicción')
            ax2.set_ylabel('Real')
            
            # Gráfico de precisión por categoría
            ax3 = fig.add_subplot(223)
            
            # Extraer precisión por categoría
            categories = []
            precision_train = []
            precision_test = []
            
            for i, category in enumerate(self.classes_):
                categories.append(category)
                precision_train.append(self.train_report[str(i)]['precision'])
                precision_test.append(self.test_report[str(i)]['precision'])
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax3.bar(x - width/2, precision_train, width, label='Entrenamiento')
            ax3.bar(x + width/2, precision_test, width, label='Prueba')
            
            ax3.set_xlabel('Categoría')
            ax3.set_ylabel('Precisión')
            ax3.set_title('Precisión por Categoría')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories, rotation=45, ha="right")
            ax3.legend()
            
            # Gráfico de sensibilidad por categoría
            ax4 = fig.add_subplot(224)
            
            # Extraer sensibilidad por categoría
            recall_train = []
            recall_test = []
            
            for i, category in enumerate(self.classes_):
                recall_train.append(self.train_report[str(i)]['recall'])
                recall_test.append(self.test_report[str(i)]['recall'])
            
            ax4.bar(x - width/2, recall_train, width, label='Entrenamiento')
            ax4.bar(x + width/2, recall_test, width, label='Prueba')
            
            ax4.set_xlabel('Categoría')
            ax4.set_ylabel('Sensibilidad')
            ax4.set_title('Sensibilidad por Categoría')
            ax4.set_xticks(x)
            ax4.set_xticklabels(categories, rotation=45, ha="right")
            ax4.legend()
            
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
        last_day = future_data["Tiempo_dias"].max()
        last_category = future_data["Category"].iloc[-1]
        last_length = future_data["Estimated_Length"].iloc[-1]
        messagebox.showinfo("Predicción Completada", 
                           f"¡Predicción completada con éxito!\n\n"
                           f"En el día {last_day}, la categoría predicha es '{last_category}' con una longitud estimada de {last_length:.2f} cm.\n\n"
                           f"Confianza en la predicción: {future_data['Probability'].iloc[-1]:.4f}")
        
        self.status_var.set(f"Predicción completada. Categoría final: {last_category}, Longitud estimada: {last_length:.2f} cm")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultinomialFishPredictor(root)
    root.mainloop()