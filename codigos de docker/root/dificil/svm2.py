import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

WINDOW = 24
TEST_WINDOW = 3 * 24
STEP = 1
TARGET = "Generacion (MWh)"

data = pd.read_csv("datos.csv")

serie = data[TARGET].values

n_total = len(serie)
n_train = int(n_total * 0.80)

train = serie[:n_train]
test = serie[n_train:]

X_train = []
y_train = []

for i in range(WINDOW, len(train)):
    X_train.append(train[i-WINDOW:i])
    y_train.append(train[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

model = SVR(
    kernel="rbf",
    C=100,
    epsilon=0.01,
    gamma="scale"
)

model.fit(X_train_scaled, y_train_scaled)

mae_list = []
rmse_list = []
r2_list = []

with open("resultados_svr.txt", "w", encoding="utf-8") as archivo:

    archivo.write(f"Total de datos: {n_total}\n")
    archivo.write(f"Entrenamiento (80%): {len(train)}\n")
    archivo.write(f"Prueba (20%): {len(test)}\n\n")

    iteracion = 1

    for inicio in range(0, len(test) - TEST_WINDOW + 1, STEP):

        test_window = test[inicio:inicio + TEST_WINDOW].copy()

        if inicio < WINDOW:
            historial = list(train[-(WINDOW-inicio):]) + list(test[:inicio])
        else:
            historial = list(test[inicio-WINDOW:inicio])

        predicciones = []

        for hora in range(TEST_WINDOW):

            x = np.array(historial[-WINDOW:]).reshape(1, -1)
            x_scaled = scaler_X.transform(x)

            pred_scaled = model.predict(x_scaled)[0]
            pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

            predicciones.append(pred)

            historial.append(pred)
            historial.pop(0)

        predicciones = np.array(predicciones)

        mae = mean_absolute_error(test_window, predicciones)
        rmse = np.sqrt(mean_squared_error(test_window, predicciones))
        r2 = r2_score(test_window, predicciones)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        archivo.write(
            f"Iteración {iteracion} | "
            f"Inicio: {inicio} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"R²: {r2:.4f}\n"
        )

        iteracion += 1

    archivo.write("\nResultados\n\n")

    archivo.write(f"Iteraciones: {len(mae_list)}\n")
    archivo.write(f"MAE promedio: {np.mean(mae_list):.4f}\n")
    archivo.write(f"RMSE promedio: {np.mean(rmse_list):.4f}\n")
    archivo.write(f"R² promedio: {np.mean(r2_list):.4f}\n\n")

    archivo.write(f"MAE std: {np.std(mae_list):.4f}\n")
    archivo.write(f"RMSE std: {np.std(rmse_list):.4f}\n")
    archivo.write(f"R² std: {np.std(r2_list):.4f}\n")
