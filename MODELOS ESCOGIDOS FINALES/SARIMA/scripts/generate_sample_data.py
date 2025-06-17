import numpy as np
import pandas as pd

# Generar datos de ejemplo para truchas
np.random.seed(42)
n = 180
t = np.arange(1, n + 1)

# Factores ambientales
T_0 = 25
O_0 = 7
C_0 = 300
pH_0 = 7.5

T = np.clip(T_0 + 3 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.5, size=n), 20, 30)
O_2 = np.clip(O_0 - 0.005 * t + np.random.normal(0, 0.2, size=n), 5, 9)
C = np.clip(C_0 + 0.3 * t + np.random.normal(0, 5, size=n), 250, 400)
pH = np.clip(pH_0 - 0.002 * t + np.random.normal(0, 0.05, size=n), 6.8, 8.2)

# Modelo Von Bertalanffy
Linf = 75
t0 = -0.5
k_base = 0.02
k = np.clip(k_base + 0.002 * (T - 25) + 0.002 * (O_2 - 7) - 0.0003 * (C - 300) + 0.003 * (pH - 7.5), 0.01, 0.05)
L = Linf * (1 - np.exp(-k * (t - t0))) + np.random.normal(0, 1, size=n)
L = np.minimum(np.maximum.accumulate(L), Linf)

datos_truchas = pd.DataFrame({
    'Tiempo_dias': t,
    'Longitud_cm': L,
    'Temperatura_C': T,
    'Oxigenacion_mg_L': O_2,
    'Conductividad_uS_cm': C,
    'pH': pH,
    'Tasa_crecimiento_k': k
})

# Generar datos de ejemplo para lechugas
dias_simulacion = 120
t_lechuga = np.arange(1, dias_simulacion + 1)

altura_max = 16
area_foliar_max = 4914.5
dias_crecimiento = 100

temperatura_l = np.clip(22 + np.sin(2 * np.pi * t_lechuga / 30) + np.random.normal(0, 0.5, dias_simulacion), 15, 30)
humedad = np.clip(70 + 10 * np.sin(2 * np.pi * t_lechuga / 25) + np.random.normal(0, 2, dias_simulacion), 50, 90)
ph_l = np.clip(6.5 + 0.1 * np.sin(2 * np.pi * t_lechuga / 40) + np.random.normal(0, 0.05, dias_simulacion), 5.5, 7.5)

def tasa_crecimiento(temp, hum, ph):
    return 0.12 - 0.003 * abs(temp - 22) - 0.002 * abs(hum - 70) - 0.01 * abs(ph - 6.5)

k_l = np.clip([tasa_crecimiento(temp, hum, ph_val) for temp, hum, ph_val in zip(temperatura_l, humedad, ph_l)], 0.02, 0.15)

t0_l = 30
altura = altura_max / (1 + np.exp(-np.array(k_l) * (t_lechuga - t0_l)))
altura = np.minimum(np.maximum.accumulate(altura), altura_max)
altura[t_lechuga > dias_crecimiento] = altura_max

area_foliar = np.clip((altura / altura_max)**2 * area_foliar_max, 0, area_foliar_max)
area_foliar = np.minimum(np.maximum.accumulate(area_foliar), area_foliar_max)
area_foliar[t_lechuga > dias_crecimiento] = area_foliar_max

datos_lechugas = pd.DataFrame({
    'Dia': t_lechuga,
    'Altura_cm': altura,
    'Area_foliar_cm2': area_foliar,
    'Temperatura_C': temperatura_l,
    'Humedad_%': humedad,
    'pH': ph_l,
    'Tasa_crecimiento_k': k_l
})

# Guardar archivos
datos_truchas.to_excel('datos_truchas_arcoiris_acuaponia_10.xlsx', index=False)
datos_lechugas.to_excel('simulacion_lechuga_realista.xlsx', index=False)

print("Archivos de datos generados exitosamente:")
print(f"Truchas: {len(datos_truchas)} registros")
print(f"Lechugas: {len(datos_lechugas)} registros")
