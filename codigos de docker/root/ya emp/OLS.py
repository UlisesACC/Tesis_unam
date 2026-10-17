import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

data = pd.read_csv("datos_ventanas.csv")

# Eliminar Timestamp si existe
if "Timestamp_Target" in data.columns:
    data = data.drop(columns=["Timestamp_Target"])

target = "Target"

X = data.drop(columns=[target])
y = data[target]

n_train = int(len(data) * 0.70)

X_train = X.iloc[:n_train]
X_test = X.iloc[n_train:]

y_train = y.iloc[:n_train]
y_test = y.iloc[n_train:]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

predicciones = model.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("\n OLS ")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(12,5))

plt.plot(y_test.values,
         label="Real")

plt.plot(predicciones,
         label="Predicho")

plt.title("OLS - Predicción vs Real")
plt.xlabel("Ventanas")
plt.ylabel("Generación (MWh)")
plt.legend()

plt.tight_layout()
plt.savefig("prediccion_vs_real_ols.png", dpi=300)
plt.show()
plt.close()