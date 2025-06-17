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
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

class PLSFishPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción Avanzada de Crecimiento de Peces - Regresión PLS")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f5f8fa")
        
        self.data = None
        self.X = None  # Variables predictoras
        self.y = None  # Variable objetivo (longitud)
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
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
        
        tk.Label(title_frame, text="Modelo de Regresión PLS (Partial Least Squares)", 
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
        
        # Columna 2: Parámetros del modelo PLS
        col2 = tk.Frame(options_frame, bg="#f5f8fa")
        col2.pack(side="left", padx=(0, 20))
        
        tk.Label(col2, text="Parámetros del modelo PLS:", bg="#f5f8fa", font=("Arial", 10)).pack(anchor="w", pady=(0, 5))
        
        # Número de componentes
        comp_frame = tk.Frame(col2, bg="#f5f8fa")
        comp_frame.pack(anchor="w", pady=(0, 5))
        
        tk.Label(comp_frame, text="Número de componentes:", bg="#f5f8fa").pack(side="left", padx=(0, 5))
        self.n_components_var = tk.StringVar(value="auto")
        tk.Entry(comp_frame, textvariable=self.n_components_var, width=8).pack(side="left")
        
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
        explanation_frame = tk.LabelFrame(summary_frame, text="Explicación del Modelo PLS", 
                                        font=("Arial", 11, "bold"), bg="#f5f8fa", fg="#2c3e50", 
                                        padx=10, pady=10)
        explanation_frame.pack(fill="x", expand=False, pady=(15, 0))
        
        explanation_text = tk.Text(explanation_frame, height=8, width=80, font=("Arial", 10), 
                                  bg="#ffffff", fg="#333333", wrap=tk.WORD)
        explanation_text.pack(fill="both", expand=True, pady=(5, 0))
        
        explanation = """La Regresión PLS (Partial Least Squares) es una técnica avanzada que combina características del análisis de componentes principales y la regresión múltiple. Es especialmente útil cuando:

1. Hay muchas variables predictoras que pueden estar correlacionadas entre sí
2. El número de observaciones es pequeño en comparación con el número de predictores

El modelo PLS funciona creando componentes (combinaciones lineales de las variables originales) que maximizan la covarianza entre las variables predictoras y la variable objetivo. Esto permite:

- Manejar eficientemente la multicolinealidad (correlación entre predictores)
- Reducir la dimensionalidad de los datos manteniendo la información relevante
- Identificar qué variables tienen mayor influencia en el crecimiento de los peces

A diferencia de otros métodos, PLS considera explícitamente la variable objetivo al crear los componentes, lo que generalmente resulta en un mejor poder predictivo."""
        
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
            
            self.status_var.set(f"Datos cargados: {len(self.data)} registros válidos")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar datos: {str(e)}")
            self.status_var.set("Error al cargar datos")
            return False
    
    def prepare_data(self):
        """Prepara los datos para el modelo PLS"""
        try:
            target_column = self.target_var.get()
            
            # Separar variables predictoras y objetivo
            self.y = self.data[target_column].values
            
            # Excluir la variable objetivo y otras columnas no relevantes
            exclude_cols = [target_column]
            self.X = self.data.drop(columns=exclude_cols).values
            
            # Guardar nombres de columnas para interpretación
            self.feature_names = self.data.drop(columns=exclude_cols).columns.tolist()
            
            # Normalizar datos si está activado
            if self.normalize_var.get():
                self.X = self.scaler_X.fit_transform(self.X)
                self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1)).ravel()
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al preparar datos: {str(e)}")
            return False
    
    def find_optimal_components(self):
        """Encuentra el número óptimo de componentes para el modelo PLS"""
        try:
            # Determinar el número máximo de componentes a probar
            max_components = min(len(self.feature_names), len(self.data) - 1)
            
            # Limitar a un máximo razonable
            max_components = min(max_components, 10)
            
            if max_components < 2:
                return 1  # Si hay muy pocos datos, usar 1 componente
            
            # Realizar validación cruzada para diferentes números de componentes
            cv_folds = int(self.cv_folds_var.get())
            
            mse_values = []
            for n_comp in range(1, max_components + 1):
                model = PLSRegression(n_components=n_comp)
                cv_scores = cross_val_score(model, self.X, self.y, 
                                          cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                          scoring='neg_mean_squared_error')
                mse_values.append(-cv_scores.mean())
                
                self.status_var.set(f"Evaluando modelo con {n_comp} componentes...")
                self.root.update()
            
            # Encontrar el número de componentes con el menor MSE
            best_n_components = np.argmin(mse_values) + 1
            
            # Crear gráfico de MSE vs número de componentes
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            
            ax.plot(range(1, max_components + 1), mse_values, 'bo-')
            ax.axvline(x=best_n_components, color='r', linestyle='--')
            ax.set_xlabel('Número de Componentes')
            ax.set_ylabel('Error Cuadrático Medio (MSE)')
            ax.set_title('Selección del Número Óptimo de Componentes')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Mostrar gráfico en una ventana emergente
            top = tk.Toplevel(self.root)
            top.title("Selección del Número Óptimo de Componentes")
            top.geometry("800x500")
            
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            return best_n_components
        except Exception as e:
            messagebox.showerror("Error", f"Error al encontrar el número óptimo de componentes: {str(e)}")
            return 2  # Valor por defecto
    
    def train_model(self):
        """Entrena el modelo PLS con los parámetros óptimos"""
        try:
            self.status_var.set("Entrenando modelo PLS...")
            self.root.update()
            
            # Determinar el número de componentes
            if self.n_components_var.get() == "auto":
                n_components = self.find_optimal_components()
            else:
                try:
                    n_components = int(self.n_components_var.get())
                    # Verificar que el número de componentes sea válido
                    max_components = min(len(self.feature_names), len(self.data) - 1)
                    if n_components > max_components:
                        n_components = max_components
                        messagebox.showinfo("Información", 
                                          f"El número de componentes se ha ajustado a {n_components} debido a limitaciones de los datos.")
                except:
                    n_components = 2  # Valor por defecto
            
            # Dividir datos en entrenamiento y prueba
            test_size = float(self.test_size_var.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42)
            
            # Entrenar modelo PLS
            self.model = PLSRegression(n_components=n_components)
            self.model.fit(X_train, y_train)
            
            # Guardar datos de entrenamiento y prueba
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            
            # Calcular predicciones en datos de entrenamiento y prueba
            self.y_train_pred = self.model.predict(X_train).ravel()
            self.y_test_pred = self.model.predict(X_test).ravel()
            
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
            
            # Calcular importancia de variables (VIP scores)
            self.calculate_vip_scores()
            
            self.status_var.set("Modelo PLS entrenado con éxito")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {str(e)}")
            self.status_var.set("Error al entrenar el modelo")
            return False
    
    def calculate_vip_scores(self):
        """Calcula los VIP scores (Variable Importance in Projection) para las variables"""
        try:
            # Obtener los pesos (weights) del modelo PLS
            W = self.model.x_weights_
            
            # Obtener los loadings de Y
            Q = self.model.y_loadings_
            
            # Calcular la suma de cuadrados de los loadings de Y
            SS_Y_explained = np.sum(Q ** 2, axis=0)
            
            # Calcular la contribución de cada componente a la varianza explicada
            var_explained_per_component = SS_Y_explained / np.sum(SS_Y_explained)
            
            # Calcular los VIP scores
            p = W.shape[0]  # Número de variables
            vip_scores = np.zeros(p)
            
            for j in range(p):
                weight_sums = np.sum([(W[j, a] ** 2) * var_explained_per_component[a] for a in range(W.shape[1])])
                vip_scores[j] = np.sqrt(p * weight_sums)
            
            # Guardar los VIP scores
            self.vip_scores = vip_scores
            
            return True
        except Exception as e:
            print(f"Error al calcular VIP scores: {str(e)}")
            self.vip_scores = np.ones(len(self.feature_names))
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
                y_pred = self.model.predict(X_pred).ravel()[0]
                
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
                
                # Guardar predicción
                predictions.append(y_pred)
                
                # Actualizar el último registro con la predicción
                last_record[target_column] = y_pred
            
            # Añadir predicciones al dataframe
            future_data[target_column] = predictions
            
            # Calcular tasas de crecimiento
            future_data["Growth_Rate"] = 0.0
            future_data["Growth_Rate_Pct"] = 0.0
            
            for i in range(len(future_data)):
                if i == 0:
                    prev_length = self.data[target_column].iloc[-1]
                else:
                    prev_length = future_data[target_column].iloc[i-1]
                
                current_length = future_data[target_column].iloc[i]
                growth = current_length - prev_length
                
                future_data.loc[future_data.index[i], "Growth_Rate"] = growth
                
                if prev_length > 0:
                    future_data.loc[future_data.index[i], "Growth_Rate_Pct"] = (growth / prev_length) * 100
            
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
        
        self.metrics_text.insert(tk.END, metrics_str)
    
    def display_parameters(self):
        """Muestra los parámetros del modelo en la pestaña de resumen"""
        # Limpiar texto anterior
        self.params_text.delete(1.0, tk.END)
        
        # Formatear parámetros
        params_str = "PARÁMETROS DEL MODELO PLS\n"
        params_str += "=" * 40 + "\n\n"
        
        params_str += f"Número de componentes: {self.model.n_components}\n"
        params_str += f"Número de variables predictoras: {len(self.feature_names)}\n"
        params_str += f"Tamaño del conjunto de entrenamiento: {len(self.y_train)} registros\n"
        params_str += f"Tamaño del conjunto de prueba: {len(self.y_test)} registros\n\n"
        
        params_str += "Variables predictoras utilizadas:\n"
        for i, feature in enumerate(self.feature_names):
            params_str += f"- {feature}\n"
        
        params_str += "\nVarianza explicada por componente:\n"
        for i in range(self.model.n_components):
            x_var = np.var(self.model.x_scores_[:, i]) / np.var(self.X_train.sum())
            y_var = np.var(self.model.y_scores_[:, i]) / np.var(self.y_train.sum())
            params_str += f"Componente {i+1}: {x_var:.2%} (X), {y_var:.2%} (Y)\n"
        
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
        ax1.set_title('Predicción de Crecimiento - Modelo PLS')
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
        
        # Ordenar variables por importancia
        indices = np.argsort(self.vip_scores)[::-1]
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_vip = self.vip_scores[indices]
        
        # Graficar importancia de variables
        bars = ax.barh(range(len(sorted_features)), sorted_vip, align='center', alpha=0.7)
        
        # Colorear barras según importancia
        for i, bar in enumerate(bars):
            if sorted_vip[i] > 1.0:
                bar.set_color('green')
            else:
                bar.set_color('orange')
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Puntuación VIP (Variable Importance in Projection)')
        ax.set_title('Importancia de Variables en el Modelo PLS')
        
        # Añadir línea de referencia en VIP = 1
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
        
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
        columns = ("Variable", "Coeficiente", "VIP Score", "Importancia")
        coef_tree = ttk.Treeview(coef_frame_inner, columns=columns, show="headings", height=15)
        
        # Configurar encabezados
        for col in columns:
            coef_tree.heading(col, text=col)
            coef_tree.column(col, width=150, anchor="center")
        
        # Obtener coeficientes del modelo
        coefs = self.model.coef_.flatten()
        
        # Añadir datos a la tabla
        for i, feature in enumerate(self.feature_names):
            # Determinar importancia
            if self.vip_scores[i] > 1.0:
                importancia = "Alta"
                tag = "alta"
            elif self.vip_scores[i] > 0.8:
                importancia = "Media"
                tag = "media"
            else:
                importancia = "Baja"
                tag = "baja"
            
            # Insertar fila en la tabla
            item_id = coef_tree.insert("", "end", values=(
                feature,
                f"{coefs[i]:.6f}",
                f"{self.vip_scores[i]:.6f}",
                importancia
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
    
    def display_diagnostics_plot(self):
        """Muestra los gráficos de diagnóstico del modelo"""
        # Limpiar gráficos anteriores
        for widget in self.diagnostics_plot_frame.winfo_children():
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
            
            # Histograma de residuos (entrenamiento)
            ax5 = fig.add_subplot(325)
            ax5.hist(self.train_residuals, bins=20, color='blue', alpha=0.7)
            ax5.set_xlabel('Residuos')
            ax5.set_ylabel('Frecuencia')
            ax5.set_title('Histograma de Residuos (Entrenamiento)')
            ax5.grid(True, linestyle='--', alpha=0.7)
            
            # Histograma de residuos (prueba)
            ax6 = fig.add_subplot(326)
            ax6.hist(self.test_residuals, bins=20, color='green', alpha=0.7)
            ax6.set_xlabel('Residuos')
            ax6.set_ylabel('Frecuencia')
            ax6.set_title('Histograma de Residuos (Prueba)')
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
    app = PLSFishPredictor(root)
    root.mainloop()