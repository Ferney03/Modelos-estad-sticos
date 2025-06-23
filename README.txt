*** Modelos de Regresión y Series Temporales ***

1. Se encuentran 3 carpetas, la primera son documentos y textos donde se respalda la regresión lineal múltiple usando el modelo de Von, la segunda están todos los modelos para simular en Python, versión 3.10 y la tercera carpeta los dos modelos escogidos (Lineal Múltiple en Regresión y Sarima en Series Temporales).

2. Para probar los modelos simulados, hay dos archivos .xlsx tanto de Truchas como de Lechugas para poder hacer las respectivas validaciones.

3. Si gustan simular otros datos con otros días, también están los simuladores en Python, un simulador para truchas y otro para Lechugas.

4. Por último están las métricas explicadas en dos documentos (Word, txt) del calculo de los errores para cada simulador.

< Dependencias generales a instalar para simular la carpeta Modelos En Python a Simular >
- Instalar Python 3.10
- pip install pandas numpy matplotlib statsmodels scikit-learn openpyxl xlrd pmdarima scipy
- Para correr el archivo con el comando py ruta_archivo o directamente desde el runner integrado en VSCode

< Dependencias generales a instalar para simular la carpeta Modelos Escogidos Finales >
- pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn plotly openpyxl xlrd pmdarima statsmodels
- Para correr el archivo de REGRESION LINEAL + VON se debe de correr la siguiente ruta:
	* PS C:\Users\Usuario\Desktop\MODELOS MATEMATICOS\Modelos Escogidos Finales\REGRESIÓN LINEAL + VON> streamlit run app.py  
- Para correr el archivo de SARIMA se debe de correr con la siguiente ruta:
	PS C:\Users\Usuario\Desktop\MODELOS MATEMATICOS\Modelos Escogidos Finales\SARIMA> streamlit run app_sarima.py 

(Al simular estos dos, les abrirá en modo web la aplicación a simular, deben de cargar los dos Datasets .xlsx para visualizar las predicciones)
(Asegurarse de cambiar o buscar la ruta adecuada donde lo descargaron)