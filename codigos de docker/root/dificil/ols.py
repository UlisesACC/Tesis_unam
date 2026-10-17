import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

TRAIN_DAYS = 2
TEST_DAYS = 3

TRAIN_HOURS = TRAIN_DAYS * 24
TEST_HOURS = TEST_DAYS * 24

WINDOW_SIZE = 24

data = pd.read_csv("datos_backtesting.csv")

X = data.drop(
    columns=[
        "Window_ID",
        "Start_Index",
        "Timestamp_Target",
        "Target"
    ]
)

y = data["Target"]

n_train = TRAIN_HOURS - WINDOW_SIZE

X_train = X.iloc[:n_train].copy()
y_train = y.iloc[:n_train].copy()

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = LinearRegression()

model.fit(X_train, y_train)

print("Modelo entrenado")

reales_global = []
pred_global = []

mae_lista = []
rmse_lista = []
r2_lista = []

inicio_prueba = n_train

while True:

    fin_prueba = inicio_prueba + TEST_HOURS

    if fin_prueba >= len(data):
        break

    prueba = data.iloc[inicio_prueba:fin_prueba].copy()

    reales = []
    predichos = []

    prueba_modificada = prueba.copy()

    print(
        f"\nEvaluando desde ventana "
        f"{inicio_prueba} "
        f"hasta "
        f"{fin_prueba}"
    )

    for k in range(len(prueba_modificada)):

        fila = prueba_modificada.iloc[k].copy()

        x = fila.drop(
            [
                "Window_ID",
                "Start_Index",
                "Timestamp_Target",
                "Target"
            ]
        ).values.reshape(1, -1)

        x = scaler.transform(x)

        pred = model.predict(x)[0]

        real = fila["Target"]

        reales.append(real)
        predichos.append(pred)

        reales_global.append(real)
        pred_global.append(pred)

        if k == len(prueba_modificada) - 1:
            continue

        siguiente = prueba_modificada.iloc[k + 1].copy()

        for t in range(1, WINDOW_SIZE):

            siguiente[f"Generacion (MWh)_t{t}"] = siguiente[
                f"Generacion (MWh)_t{t+1}"
            ]

        siguiente[f"Generacion (MWh)_t{WINDOW_SIZE}"] = pred

        prueba_modificada.iloc[k + 1] = siguiente
    reales = np.array(reales)
    predichos = np.array(predichos)

    mae = mean_absolute_error(reales, predichos)
    rmse = np.sqrt(mean_squared_error(reales, predichos))
    r2 = r2_score(reales, predichos)

    mae_lista.append(mae)
    rmse_lista.append(rmse)
    r2_lista.append(r2)

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    inicio_prueba += 1

reales_global = np.array(reales_global)
pred_global = np.array(pred_global)

print("\n==============================")
print("RESULTADOS PROMEDIO")
print("==============================")
print(f"Ventanas evaluadas : {len(mae_lista)}")
print(f"MAE promedio       : {np.mean(mae_lista):.4f}")
print(f"RMSE promedio      : {np.mean(rmse_lista):.4f}")
print(f"R² promedio        : {np.mean(r2_lista):.4f}")

print("\n==============================")
print("RESULTADOS GLOBALES")
print("==============================")
print(f"MAE : {mean_absolute_error(reales_global,pred_global):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(reales_global,pred_global)):.4f}")
print(f"R²  : {r2_score(reales_global,pred_global):.4f}")

plt.figure(figsize=(14,6))

plt.plot(
    reales_global,
    label="Real"
)

plt.plot(
    pred_global,
    label="Predicho"
)

plt.xlabel("Horas pronosticadas")
plt.ylabel("Generación (MWh)")
plt.title("Backtesting OLS (Entrenamiento único)")
plt.legend()

plt.tight_layout()

plt.savefig(
    "backtesting_ols.png",
    dpi=300
)

plt.show()
plt.close()