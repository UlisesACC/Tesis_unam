import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

if len(data_70) < history + train_window + 2:
    raise ValueError(
        "Se requieren al menos 50 registros en el primer 70 % "
        "para entrenar y evaluar al menos dos predicciones."
    )

serie = pd.to_numeric(data_70[target], errors="raise").to_numpy(dtype=float)
if not np.isfinite(serie).all():
    raise ValueError("La generación contiene valores faltantes o no finitos.")

if data_70["fecha_hora"].duplicated().any():
    raise ValueError("Hay fechas duplicadas en la serie seleccionada.")

intervalos = data_70["fecha_hora"].diff().dropna()
if intervalos.nunique() != 1:
    raise ValueError(
        "Los registros no tienen un intervalo temporal constante. "
        "Revise las fechas y los registros faltantes."
    )

# Cada fila contiene 24 generaciones anteriores; su respuesta es la siguiente.
X = np.array([
    serie[i - history:i]
    for i in range(history, len(serie))
])
y = serie[history:]
fechas_objetivo = data_70["fecha_hora"].iloc[history:].reset_index(drop=True)

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
    y_pred = model.predict(X_test_scaled)[0]

    reales.append(y[i])
    predichos.append(y_pred)
    fechas.append(fechas_objetivo.iloc[i])

reales = np.array(reales)
predichos = np.array(predichos)

mae = mean_absolute_error(reales, predichos)
rmse = np.sqrt(mean_squared_error(reales, predichos))
r2 = r2_score(reales, predichos)

print("\nOLS – Ventana móvil univariante")
print(f"Área: {area_seleccionada}")
print(f"Intervalo entre registros: {intervalos.iloc[0]}")
print(f"Historia de entrada: {history} registros de generación")
print(f"Ejemplos por ajuste: {train_window}")
print(f"Periodo evaluado: {fechas[0]} a {fechas[-1]}")
print(f"Predicciones evaluadas: {len(reales)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(fechas, reales, label="Real", color="blue")
plt.plot(fechas, predichos, label="Predicho", color="orange")
plt.xlabel("Fecha y hora")
plt.ylabel("Generación (MWh)")
plt.title("OLS – Generación a partir de 24 generaciones anteriores")
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("prediccion_vs_real_ols.png", dpi=300, bbox_inches="tight")
plt.close()
