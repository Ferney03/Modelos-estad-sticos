import numpy as np
import pandas as pd

# Parámetros iniciales
np.random.seed(42)
n = 180  # Días de observación
t = np.arange(1, n + 1)  # Días normales

# Factores ambientales dentro de rangos realistas
T_0 = 25  # Temperatura base (°C)
O_0 = 7  # Oxígeno base (mg/L)
C_0 = 300  # Conductividad base (uS/cm)
pH_0 = 7.5  # pH base

# Modelando fluctuaciones ambientales
T = np.clip(T_0 + 3 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.5, size=n), 20, 30)
O_2 = np.clip(O_0 - 0.005 * t + np.random.normal(0, 0.2, size=n), 5, 9)
C = np.clip(C_0 + 0.3 * t + np.random.normal(0, 5, size=n), 250, 400)
pH = np.clip(pH_0 - 0.002 * t + np.random.normal(0, 0.05, size=n), 6.8, 8.2)

# Parámetros de Von Bertalanffy ajustados
Linf = 75  # Longitud máxima teórica en cm para trucha arco iris
t0 = -0.5  # Día teórico inicial
k_base = 0.02  # Ajustar k en función de condiciones ambientales
k = np.clip(k_base + 0.002 * (T - 25) + 0.002 * (O_2 - 7) - 0.0003 * (C - 300) + 0.003 * (pH - 7.5), 0.01, 0.05)

# Modelo de crecimiento
L = Linf * (1 - np.exp(-k * (t - t0))) + np.random.normal(0, 1, size=n)

# Asegurar crecimiento progresivo sin valores negativos y no sobrepasar los 75 cm
L = np.minimum(np.maximum.accumulate(L), Linf)

# Crear DataFrame
datos_temporales = pd.DataFrame({
    'Tiempo_dias': t,
    'Longitud_cm': L,
    'Temperatura_C': T,
    'Oxigenacion_mg_L': O_2,
    'Conductividad_uS_cm': C,
    'pH': pH,
    'Tasa_crecimiento_k': k
})

# Guardar datos
datos_temporales.to_excel('Dataset_Truchas_180_dias.xlsx', index=False)

print(datos_temporales.head())