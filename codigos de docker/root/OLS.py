
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

data = pd.read_csv("datos.csv")

data["fecha_hora"] = pd.to_datetime(
    data["Timestamp"],
    format="%d/%m/%Y %H:%M",
    dayfirst=True
)

fecha_inicio = pd.Timestamp("2022-01-01")
fecha_fin_exclusiva = pd.Timestamp("2023-01-01")
area_seleccionada = data["Area"].unique()[0]
target = "Generacion (MWh)"

data_periodo = data[
    (data["fecha_hora"] >= fecha_inicio)
    & (data["fecha_hora"] < fecha_fin_exclusiva)
    & (data["Area"] == area_seleccionada)
].copy()

data_periodo = data_periodo.sort_values(
    "fecha_hora", kind="stable"
).reset_index(drop=True)

n_70 = int(len(data_periodo) * 0.7)
data_70 = data_periodo.iloc[:n_70].copy()

history = 24
train_window = 24
intervalo_esperado = pd.Timedelta(hours=1)

if len(data_70) < history + train_window + 2:
    raise ValueError(
        "No hay suficientes registros para entrenar el modelo."
    )

data_70[target] = pd.to_numeric(
    data_70[target], errors="raise"
)

if not np.isfinite(data_70[target].to_numpy(dtype=float)).all():
    raise ValueError(
        "La generación contiene valores faltantes o no finitos."
    )

if data_70["fecha_hora"].duplicated().any():
    raise ValueError(
        "Existen fechas duplicadas en los datos."
    )

serie = data_70[target].to_numpy(dtype=float)
fechas_serie = data_70["fecha_hora"]

intervalos = fechas_serie.diff().dropna()

print("\nInformación de los datos")
print(f"Área seleccionada: {area_seleccionada}")
print(f"Registros del periodo: {len(data_periodo)}")
print(f"Registros del primer 70 %: {len(data_70)}")
print(f"Intervalo mínimo: {intervalos.min()}")
print(f"Intervalo máximo: {intervalos.max()}")
print(
    "Saltos distintos de una hora:",
    int((intervalos != intervalo_esperado).sum())
)

# Construcción de ventanas con 24 horas consecutivas
X = []
y = []
fechas_objetivo = []

for i in range(history, len(serie)):

    fechas_ventana = fechas_serie.iloc[
        i - history:i + 1
    ]

    diferencias = fechas_ventana.diff().dropna()

    if not (diferencias == intervalo_esperado).all():
        continue

    X.append(serie[i - history:i])
    y.append(serie[i])
    fechas_objetivo.append(fechas_serie.iloc[i])

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)
fechas_objetivo = pd.DatetimeIndex(fechas_objetivo)

if len(X) < train_window + 2:
    raise ValueError(
        "No hay suficientes ventanas de 24 horas consecutivas "
        "para entrenar y evaluar el modelo."
    )

# Predicción mediante ventana móvil
reales = []
predichos = []
fechas = []

for i in range(train_window, len(X)):

    X_train = X[i - train_window:i]
    y_train = y[i - train_window:i]
    X_test = X[i:i + 1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    prediccion = model.predict(X_test_scaled)[0]

    reales.append(y[i])
    predichos.append(prediccion)
    fechas.append(fechas_objetivo[i])

reales = np.asarray(reales)
predichos = np.asarray(predichos)
fechas = pd.DatetimeIndex(fechas)

# Métricas
mae = mean_absolute_error(reales, predichos)
rmse = np.sqrt(mean_squared_error(reales, predichos))
r2 = r2_score(reales, predichos)

print("\nOLS - Ventana móvil univariante")
print(f"Área: {area_seleccionada}")
print(f"Historia de entrada: {history} horas")
print(f"Ejemplos por ajuste: {train_window}")
print(f"Ventanas válidas: {len(X)}")
print(f"Predicciones evaluadas: {len(reales)}")
print(f"Periodo evaluado: {fechas[0]} a {fechas[-1]}")

print("\nMétricas")
print(f"MAE: {mae:.4f} MWh")
print(f"RMSE: {rmse:.4f} MWh")
print(f"R²: {r2:.4f}")

# Resultados
resultados = pd.DataFrame({
    "fecha_hora": fechas,
    "real": reales,
    "predicho": predichos
})

resultados["error"] = (
    resultados["real"] - resultados["predicho"]
)

resultados.to_csv(
    "resultados_ols.csv",
    index=False
)

# Gráfica del periodo completo
plt.figure(figsize=(12, 5))

plt.plot(
    fechas, reales,
    label="Real",
    color="blue",
    linewidth=1
)

plt.plot(
    fechas, predichos,
    label="Predicho",
    color="orange",
    linewidth=1
)

plt.xlabel("Fecha y hora")
plt.ylabel("Generación (MWh)")
plt.title("OLS - Predicción de generación")
plt.legend()
plt.grid(alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.savefig(
    "prediccion_vs_real_ols.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Gráfica ampliada de 3 días
inicio_zoom = fechas[0]
fin_zoom = inicio_zoom + pd.Timedelta(days=3)

mascara_zoom = (
    (fechas >= inicio_zoom)
    & (fechas < fin_zoom)
)

fechas_zoom = fechas[mascara_zoom]
reales_zoom = reales[mascara_zoom]
predichos_zoom = predichos[mascara_zoom]

if len(fechas_zoom) == 0:
    raise ValueError("No hay predicciones en el periodo ampliado.")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    fechas_zoom,
    reales_zoom,
    label="Generación real",
    color="blue",
    marker="o",
    markersize=3,
    linewidth=1.5
)

ax.plot(
    fechas_zoom,
    predichos_zoom,
    label="Generación predicha",
    color="orange",
    marker=".",
    markersize=4,
    linewidth=1.5
)

ax.set_xlim(inicio_zoom, fin_zoom)

ax.xaxis.set_major_locator(
    mdates.HourLocator(interval=6)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d/%m %H:%M")
)

ax.set_xlabel("Fecha y hora")
ax.set_ylabel("Generación (MWh)")
ax.set_title(
    "OLS - Comparación de generación real y predicha (3 días)"
)

ax.legend()
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "prediccion_vs_real_ols_3dias.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nArchivos generados:")
print("resultados_ols.csv")
print("prediccion_vs_real_ols.png")
print("prediccion_vs_real_ols_3dias.png")
