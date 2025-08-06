import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import warnings

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore')

def cargar_y_procesar_csvs(directorio_csvs=".", patron_archivos="*.csv"):
    """
    Carga y procesa múltiples CSVs extrayendo estadísticas min/max
    
    Args:
        directorio_csvs (str): Directorio donde están los CSVs
        patron_archivos (str): Patrón para buscar archivos CSV
    
    Returns:
        dict: Diccionario con todas las estadísticas
    """
    
    # Columnas numéricas que queremos analizar
    columnas_numericas = [
        'Temperature (C)',
        'Turbidity(NTU)', 
        'Dissolved Oxygen(g/ml)',
        'PH',
        'Ammonia(g/ml)',
        'Nitrate(g/ml)',
        'Population',
        'Fish_Length(cm)',
        'Fish_Weight(g)'
    ]
    
    # Buscar todos los archivos CSV
    ruta_busqueda = os.path.join(directorio_csvs, patron_archivos)
    archivos_csv = glob.glob(ruta_busqueda)
    
    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en: {ruta_busqueda}")
        return None
    
    print(f"📁 Encontrados {len(archivos_csv)} archivos CSV")
    print("-" * 60)
    
    # Diccionarios para almacenar resultados
    stats_por_archivo = {}
    datos_globales = []
    total_filas_global = 0
    
    # Procesar cada CSV
    for i, archivo in enumerate(archivos_csv, 1):
        nombre_archivo = os.path.basename(archivo)
        print(f"🔄 Procesando {i}/{len(archivos_csv)}: {nombre_archivo}")
        
        try:
            # Leer CSV
            df = pd.read_csv(archivo)
            total_filas = len(df)
            total_filas_global += total_filas
            
            print(f"   📊 Filas cargadas: {total_filas:,}")
            
            # Limpiar nombres de columnas (quitar espacios)
            df.columns = df.columns.str.strip()
            
            # Verificar qué columnas numéricas existen
            columnas_existentes = [col for col in columnas_numericas if col in df.columns]
            
            if not columnas_existentes:
                print(f"   ⚠️  No se encontraron columnas numéricas conocidas")
                continue
            
            # Convertir a numérico y manejar errores
            for col in columnas_existentes:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calcular estadísticas para este archivo
            stats_archivo = {
                'archivo': nombre_archivo,
                'total_filas': total_filas,
                'estadisticas': {}
            }
            
            for col in columnas_existentes:
                # Filtrar valores no nulos
                datos_validos = df[col].dropna()
                
                if len(datos_validos) > 0:
                    stats_archivo['estadisticas'][col] = {
                        'min': float(datos_validos.min()),
                        'max': float(datos_validos.max()),
                        'mean': float(datos_validos.mean()),
                        'datos_validos': len(datos_validos)
                    }
                    
                    # Agregar a datos globales para cálculo general
                    datos_globales.extend([(col, val) for val in datos_validos])
            
            stats_por_archivo[nombre_archivo] = stats_archivo
            
            print(f"   ✅ Procesado exitosamente")
            
        except Exception as e:
            print(f"   ❌ Error procesando {nombre_archivo}: {str(e)}")
            continue
    
    # Calcular estadísticas globales
    print("\n" + "="*60)
    print("📈 CALCULANDO ESTADÍSTICAS GLOBALES...")
    
    stats_globales = {}
    
    # Organizar datos por columna
    datos_por_columna = {}
    for col, valor in datos_globales:
        if col not in datos_por_columna:
            datos_por_columna[col] = []
        datos_por_columna[col].append(valor)
    
    # Calcular min/max globales
    for col, valores in datos_por_columna.items():
        if valores:
            stats_globales[col] = {
                'min_global': float(min(valores)),
                'max_global': float(max(valores)),
                'mean_global': float(np.mean(valores)),
                'total_datos': len(valores)
            }
    
    # Compilar resultado final
    resultado = {
        'resumen': {
            'archivos_procesados': len(stats_por_archivo),
            'total_filas_global': total_filas_global,
            'columnas_analizadas': list(stats_globales.keys())
        },
        'estadisticas_globales': stats_globales,
        'estadisticas_por_archivo': stats_por_archivo
    }
    
    return resultado

def mostrar_resultados(resultado):
    """Muestra los resultados de forma organizada"""
    
    if not resultado:
        return
    
    print("\n" + "="*80)
    print("📋 RESUMEN GENERAL")
    print("="*80)
    
    resumen = resultado['resumen']
    print(f"Archivos procesados: {resumen['archivos_procesados']}")
    print(f"Total de filas: {resumen['total_filas_global']:,}")
    print(f"Columnas analizadas: {len(resumen['columnas_analizadas'])}")
    
    print("\n" + "="*80)
    print("🌍 ESTADÍSTICAS GLOBALES (TODOS LOS ARCHIVOS)")
    print("="*80)
    
    stats_globales = resultado['estadisticas_globales']
    
    for columna, stats in stats_globales.items():
        print(f"\n📊 {columna}:")
        print(f"   Mínimo global: {stats['min_global']:.4f}")
        print(f"   Máximo global: {stats['max_global']:.4f}")
        print(f"   Promedio global: {stats['mean_global']:.4f}")
        print(f"   Total datos: {stats['total_datos']:,}")
    
    print("\n" + "="*80)
    print("📁 ESTADÍSTICAS POR ARCHIVO")
    print("="*80)
    
    for nombre_archivo, data in resultado['estadisticas_por_archivo'].items():
        print(f"\n🔹 {nombre_archivo}")
        print(f"   Filas: {data['total_filas']:,}")
        
        for col, stats in data['estadisticas'].items():
            print(f"   {col}: min={stats['min']:.4f}, max={stats['max']:.4f}")

def guardar_resultados_csv(resultado, archivo_salida="estadisticas_resultado.csv"):
    """Guarda los resultados en un CSV"""
    
    if not resultado:
        return
    
    # Crear DataFrame con estadísticas globales
    filas_globales = []
    for col, stats in resultado['estadisticas_globales'].items():
        filas_globales.append({
            'Tipo': 'Global',
            'Archivo': 'TODOS',
            'Columna': col,
            'Minimo': stats['min_global'],
            'Maximo': stats['max_global'],
            'Promedio': stats['mean_global'],
            'Total_Datos': stats['total_datos']
        })
    
    # Crear DataFrame con estadísticas por archivo
    filas_archivo = []
    for nombre_archivo, data in resultado['estadisticas_por_archivo'].items():
        for col, stats in data['estadisticas'].items():
            filas_archivo.append({
                'Tipo': 'Por_Archivo',
                'Archivo': nombre_archivo,
                'Columna': col,
                'Minimo': stats['min'],
                'Maximo': stats['max'],
                'Promedio': stats['mean'],
                'Total_Datos': stats['datos_validos']
            })
    
    # Combinar y guardar
    todas_filas = filas_globales + filas_archivo
    df_resultado = pd.DataFrame(todas_filas)
    df_resultado.to_csv(archivo_salida, index=False)
    
    print(f"\n💾 Resultados guardados en: {archivo_salida}")

# FUNCIÓN PRINCIPAL
def main():
    """Función principal del script"""
    
    print("🚀 INICIANDO PROCESAMIENTO DE CSVs")
    print("="*60)
    
    # Configuración - CAMBIAR ESTAS RUTAS SEGÚN TUS NECESIDADES
    directorio = "."  # Directorio actual, cambiar si los CSVs están en otra carpeta
    patron = "*.csv"  # Buscar todos los .csv
    
    # Procesar archivos
    resultado = cargar_y_procesar_csvs(directorio, patron)
    
    if resultado:
        # Mostrar resultados en pantalla
        mostrar_resultados(resultado)
        
        # Guardar resultados en CSV
        guardar_resultados_csv(resultado)
        
        print("\n🎉 ¡PROCESAMIENTO COMPLETADO!")
    else:
        print("\n❌ No se pudo completar el procesamiento")

# EJECUTAR EL SCRIPT
if __name__ == "__main__":
    main()