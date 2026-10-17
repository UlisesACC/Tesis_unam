import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
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

data["Hour_of_Day"] = data["Hour"] + data["Minute"] / 60
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

target = "Generacion (MWh)"
data_periodo["Gen_lag1"] = data_periodo[target].shift(1)
data_periodo = data_periodo.dropna()

features = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday",
    "Gen_lag1"
]

n_70 = int(len(data_periodo) * 0.7)
data_70 = data_periodo.iloc[:n_70]

window_size = 13
reales = []
predichos = []

for i in range(len(data_70) - window_size + 1):
    ventana = data_70.iloc[i:i + window_size]

    train = ventana.iloc[:-1]
    test = ventana.iloc[-1:]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    model = ElasticNetCV(
        alphas=[0.001, 0.01, 0.1, 1, 10],
        l1_ratio=[0.1, 0.5, 0.9],
        cv=3
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    reales.append(y_test.values[0])
    predichos.append(y_pred)

reales = np.array(reales)
predichos = np.array(predichos)

mae = mean_absolute_error(reales, predichos)
rmse = np.sqrt(mean_squared_error(reales, predichos))
r2 = r2_score(reales, predichos)

plt.figure(figsize=(8, 5))
plt.plot(reales, label="Real")
plt.plot(predichos, label="Predicho")
plt.xlabel("Horas")
plt.ylabel("(MWh)")
plt.title("Elastic Net – Predicción vs Real")
plt.legend()
plt.savefig("prediccion_vs_real_elasticnet.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")