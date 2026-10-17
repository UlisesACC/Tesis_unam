import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D

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

features = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday"
]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_all = data_periodo[features].values
y_all = data_periodo[target].values.reshape(-1, 1)

X_scaled = scaler_X.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all)

window_size = 13

X_seq = []
y_seq = []

for i in range(len(X_scaled) - window_size):
    X_seq.append(X_scaled[i:i + window_size])
    y_seq.append(y_scaled[i + window_size])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

def build_transformer(input_shape):
    inputs = Input(shape=input_shape)

    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = Dropout(0.1)(x)
    x = LayerNormalization()(x + inputs)

    x_ff = Dense(64, activation="relu")(x)
    x_ff = Dense(input_shape[-1])(x_ff)
    x = LayerNormalization()(x + x_ff)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model

model = build_transformer((window_size, len(features)))

history = model.fit(
    X_seq,
    y_seq,
    epochs=50,
    batch_size=32,
    verbose=1
)

y_pred_scaled = model.predict(X_seq, verbose=0)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_real = scaler_y.inverse_transform(y_seq)

mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)


plt.figure(figsize=(8, 5))
plt.plot(y_real, label="Real")
plt.plot(y_pred, label="Predicho")
plt.xlabel("Tiempo")
plt.ylabel("(MWh)")
plt.title("Transformer – Predicción vs Real")
plt.legend()
plt.savefig("prediccion_vs_real_transformer.png", dpi=300, bbox_inches="tight")
plt.close()

archivo = "resultados.txt"

modo = "a" if os.path.exists(archivo) else "w"

with open(archivo, modo) as f:
    f.write("\n")
    f.write("MODELO: TRANSFORMER\n")
    f.write("\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R²: {r2:.2f}\n")

print("Entrenamiento finalizado. Resultados guardados en resultados.txt")


plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], color="blue", linewidth=2)
plt.title("Curva de pérdida (Loss)")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.grid(True)

plt.savefig("loss_transformer.png", dpi=300, bbox_inches="tight")
plt.close()