import numpy as np
import pandas as pd

# Parámetros iniciales del sistema
np.random.seed(42)
n = 180  # Días de observación

# Tiempo en días
t = np.linspace(0, 180, n)

# Variaciones temporales de las variables ambientales
T_0 = 25  # Temperatura base
O_0 = 7   # Oxígeno base
C_0 = 300 # Conductividad base
pH_0 = 7.5  # pH base

# Modelando fluctuaciones ambientales
T = T_0 + 3 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.5, size=n)  # Variación estacional
O_2 = O_0 - 0.01 * t + 1 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.2, size=n)  # Oxígeno con tendencia
C = C_0 + 0.5 * t + np.random.normal(0, 10, size=n)  # Conductividad creciente
pH = pH_0 - 0.005 * t + np.random.normal(0, 0.1, size=n)  # pH con tendencia a bajar

# Parámetros de Von Bertalanffy con k dependiente del tiempo
Linf = 35  # Longitud máxima teórica
t0 = -0.5  # Día teórico inicial
k = 0.02 + 0.002 * (T - 25) + 0.003 * (O_2 - 7) - 0.0005 * (C - 300) + 0.005 * (pH - 7.5)  # k dinámico

# Aplicando el modelo de Von Bertalanffy con k variable
L = Linf * (1 - np.exp(-k * (t - t0))) + np.random.normal(0, 0.5, size=n)  # Agregando ruido

# Crear DataFrame con los datos simulados
datos_temporales = pd.DataFrame({
    'Tiempo_dias': t,
    'Longitud_cm': L,
    'Temperatura_C': T,
    'Oxigenacion_mg_L': O_2,
    'Conductividad_uS_cm': C,
    'pH': pH,
    'Tasa_crecimiento_k': k
})
datos_temporales.to_excel('datos_sinteticos_acuaponia.xlsx')
