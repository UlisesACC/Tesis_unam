import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

data = pd.read_csv("datos.csv")

time = pd.to_datetime(
    data["Timestamp"],
    format="%d/%m/%Y %H:%M",
    dayfirst=True
)

data["Year"] = time.dt.year
data["Month"] = time.dt.month
data["Day"] = time.dt.day
data["Hour"] = time.dt.hour
data["Minute"] = time.dt.minute

data["fecha_hora"] = pd.to_datetime(
    data[["Year", "Month", "Day", "Hour"]]
)

data["Hour_of_Day"] = data["Hour"] + data["Minute"] / 60.0
data["Day_of_Year"] = data["fecha_hora"].dt.dayofyear
data["Weekday"] = data["fecha_hora"].dt.weekday

fecha_inicio = "2022-01-01"
fecha_fin = "2022-12-31"
area_seleccionada = data["Area"].unique()[0]

data_periodo = data[
    (data["fecha_hora"] >= fecha_inicio) &
    (data["fecha_hora"] <= fecha_fin) &
    (data["Area"] == area_seleccionada)
].copy()

data_periodo = data_periodo.sort_values("fecha_hora").reset_index(drop=True)

target = "Generacion (MWh)"

data_periodo["Gen_lag1"] = data_periodo[target].shift(1)
data_periodo = data_periodo.dropna().copy()

features = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday",
    "Gen_lag1"
]

n_total = len(data_periodo)
n_70 = int(n_total * 0.7)

data_70 = data_periodo.iloc[:n_70].copy()

horizon = 10
history = 12
window_size = history + horizon

alphas = [0.001, 0.01, 0.1, 1, 10]

reales = []
predichos = []

for i in range(len(data_70) - window_size + 1):

    ventana = data_70.iloc[i:i + window_size]

    train = ventana.iloc[:history]
    future = ventana.iloc[history:history + horizon]

    X_train = train[features]

    y_train = []
    for j in range(len(train)):
        if j + horizon < len(ventana):
            y_train.append(
                ventana[target].iloc[j+1:j+1+horizon].values
            )

    y_train = np.array(y_train)

    X_train = X_train.iloc[:len(y_train)]

    X_test = train[features].iloc[-1:]

    model = RidgeCV(alphas=alphas, cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)[0]  

    reales.append(future[target].values)
    predichos.append(y_pred)


reales = np.array(reales)
predichos = np.array(predichos)

mae = mean_absolute_error(reales, predichos)
rmse = np.sqrt(mean_squared_error(reales, predichos))
r2 = r2_score(reales, predichos)

print("\nRIDGE – Predicción Multi-output (10 pasos)")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")


plt.figure(figsize=(10, 5))

plt.plot(reales[:, 0], label="Real t+1")
plt.plot(predichos[:, 0], label="Predicho t+1")

plt.xlabel("Tiempo")
plt.ylabel("Generación (MWh)")
plt.title("Ridge Multi-output (Primer paso)")
plt.legend()

plt.tight_layout()
plt.savefig("prediccion_ridge_multioutput.png", dpi=300, bbox_inches="tight")
plt.close()