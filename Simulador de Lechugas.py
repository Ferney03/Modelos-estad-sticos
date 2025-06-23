import numpy as np
import pandas as pd

# Parámetros fijos
np.random.seed(42)
dias_simulacion = 120
t = np.arange(1, dias_simulacion + 1)

# Límites biológicos
altura_max = 16  # cm
area_foliar_max = 4914.5  # cm²
dias_crecimiento = 100  # Días para alcanzar el máximo

# Simulación de variables ambientales dentro de rangos normales
temperatura = np.clip(22 + np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.5, dias_simulacion), 15, 30)
humedad = np.clip(70 + 10 * np.sin(2 * np.pi * t / 25) + np.random.normal(0, 2, dias_simulacion), 50, 90)
ph = np.clip(6.5 + 0.1 * np.sin(2 * np.pi * t / 40) + np.random.normal(0, 0.05, dias_simulacion), 5.5, 7.5)

# Tasa de crecimiento afectada suavemente por condiciones
def tasa_crecimiento(temp, hum, ph):
    return 0.12 \
        - 0.003 * abs(temp - 22) \
        - 0.002 * abs(hum - 70) \
        - 0.01 * abs(ph - 6.5)

k = np.clip([tasa_crecimiento(temp, hum, ph_val) for temp, hum, ph_val in zip(temperatura, humedad, ph)], 0.02, 0.15)

t0 = 30  # Punto de inflexión (día más rápido de crecimiento)
altura = altura_max / (1 + np.exp(-np.array(k) * (t - t0)))

# Truncar altura al máximo una vez alcanzado
altura = np.minimum(np.maximum.accumulate(altura), altura_max)
altura[t > dias_crecimiento] = altura_max  # No crecer más allá del día 100

# Área foliar proporcional al cuadrado de la altura (modelo común)
area_foliar = np.clip((altura / altura_max)**2 * area_foliar_max, 0, area_foliar_max)
area_foliar = np.minimum(np.maximum.accumulate(area_foliar), area_foliar_max)
area_foliar[t > dias_crecimiento] = area_foliar_max  # Mantener constante

# Construir DataFrame
simulacion_lechuga = pd.DataFrame({
    'Dia': t,
    'Altura_cm': altura,
    'Area_foliar_cm2': area_foliar,
    'Temperatura_C': temperatura,
    'Humedad_%': humedad,
    'pH': ph,
    'Tasa_crecimiento_k': k
})

# Guardar resultados
simulacion_lechuga.to_excel('Dataset_Lechugas_120_dias.xlsx', index=False)

# Mostrar primeras filas
print(simulacion_lechuga.head(10))
