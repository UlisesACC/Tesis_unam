import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention,
    Dropout, GlobalAveragePooling1D, LeakyReLU
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.epoch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_losses.append(logs.get("loss"))

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

def build_transformer(input_shape, num_heads):
    inputs = Input(shape=input_shape)

    x = MultiHeadAttention(num_heads=num_heads, key_dim=16)(inputs, inputs)
    x = Dropout(0.1)(x)
    x = LayerNormalization()(x + inputs)

    x_ff = Dense(64)(x)
    x_ff = LeakyReLU(alpha=0.1)(x_ff)
    x_ff = Dense(input_shape[-1])(x_ff)

    x = LayerNormalization()(x + x_ff)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    optimizer = SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=optimizer, loss="mse")

    return model

heads_list = [2, 4, 8]

archivo = "salida.txt"
modo = "a" if os.path.exists(archivo) else "w"

for heads in heads_list:

    print(f"\nEntrenando modelo con {heads} cabezas...\n")

    model = build_transformer((window_size, len(features)), heads)

    history_cb = LossHistory()

    model.fit(
        X_seq,
        y_seq,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[history_cb]
    )

    y_pred_scaled = model.predict(X_seq, verbose=0)

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_real = scaler_y.inverse_transform(y_seq)

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)

    plt.figure(figsize=(10, 5))

    plt.plot(history_cb.batch_losses, alpha=0.3, label="Batch loss")

    plt.plot(
        np.linspace(0, len(history_cb.batch_losses), len(history_cb.epoch_losses)),
        history_cb.epoch_losses,
        linewidth=2,
        label="Epoch loss"
    )

    plt.ylim(
        min(history_cb.batch_losses) * 0.95,
        max(history_cb.batch_losses) * 1.05
    )

    plt.title(f"Loss Training ({heads} heads)")
    plt.xlabel("Iteraciones")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(f"loss_{heads}_heads.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(y_real, label="Real")
    plt.plot(y_pred, label="Predicho")
    plt.legend()
    plt.title(f"Pred vs Real ({heads} heads)")
    plt.savefig(f"pred_{heads}_heads.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(archivo, modo) as f:
        f.write("\n")
        f.write(f"TRANSFORMER - {heads} HEADS\n")
        f.write("\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

print("\nEntrenamiento completo.")