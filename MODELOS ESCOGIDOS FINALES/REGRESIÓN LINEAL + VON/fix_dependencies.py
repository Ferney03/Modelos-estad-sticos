import subprocess
import sys

def install_compatible_versions():
    print("Instalando versiones compatibles de las bibliotecas...")
    
    # Desinstalar versiones actuales que pueden estar causando conflictos
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "streamlit", "protobuf"])
    except:
        print("No se pudieron desinstalar las versiones anteriores, continuando...")
    
    # Instalar versiones específicas compatibles
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf==3.20.3"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit==1.22.0"])
    
    # Instalar el resto de dependencias
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "pandas", "numpy", "matplotlib", "seaborn", 
                          "scipy", "scikit-learn", "plotly", "openpyxl", "xlrd"])
    
    print("\n✅ Instalación completada. Ahora puedes ejecutar la aplicación con:")
    print("streamlit run app.py")

if __name__ == "__main__":
    install_compatible_versions()
