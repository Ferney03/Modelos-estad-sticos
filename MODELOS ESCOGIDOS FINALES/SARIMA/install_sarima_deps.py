import subprocess
import sys

def install_sarima_dependencies():
    print("Instalando dependencias para SARIMA...")
    
    # Lista de paquetes necesarios
    packages = [
        "streamlit==1.22.0",
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "plotly",
        "openpyxl",
        "xlrd",
        "pmdarima",
        "statsmodels",
        "scikit-learn"
    ]
    
    for package in packages:
        try:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {package}: {e}")
    
    print("\n🎉 Instalación completada!")
    print("Ejecuta la aplicación con: streamlit run app_sarima.py")

if __name__ == "__main__":
    install_sarima_dependencies()
