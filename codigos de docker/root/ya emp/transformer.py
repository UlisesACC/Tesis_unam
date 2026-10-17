import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    GlobalAveragePooling1D
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

data = pd.read_csv("datos_ventanas.csv")

if "Timestamp_Target" in data.columns:
    data = data.drop(columns=["Timestamp_Target"])

target = "Target"

feature_names = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday",
    "Generacion (MWh)"
]

window_size = 24

X = []
y = []

for _, row in data.iterrows():

    seq = []

    for t in range(1, window_size + 1):

        seq.append([
            row[f"Hour_of_Day_t{t}"],
            row[f"Day_of_Year_t{t}"],
            row[f"Year_t{t}"],
            row[f"Month_t{t}"],
            row[f"Weekday_t{t}"],
            row[f"Generacion (MWh)_t{t}"]
        ])

    X.append(seq)
    y.append(row[target])

X = np.array(X)
y = np.array(y).reshape(-1,1)

n_train = int(len(X) * 0.70)

X_train = X[:n_train]
X_test = X[n_train:]

y_train = y[:n_train]
y_test = y[n_train:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_test = scaler_X.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

y_train = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

def build_transformer(input_shape):

    inputs = Input(shape=input_shape)

    x = MultiHeadAttention(
        num_heads=2,
        key_dim=16
    )(inputs, inputs)

    x = Dropout(0.1)(x)

    x = LayerNormalization()(x + inputs)

    ff = Dense(64, activation="relu")(x)
    ff = Dense(input_shape[-1])(ff)

    x = LayerNormalization()(x + ff)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

model = build_transformer((window_size, len(feature_names)))

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=1
)

pred_scaled = model.predict(
    X_test,
    verbose=0
)

pred = scaler_y.inverse_transform(pred_scaled)
real = scaler_y.inverse_transform(y_test_scaled)

mae = mean_absolute_error(real, pred)
rmse = np.sqrt(mean_squared_error(real, pred))
r2 = r2_score(real, pred)

print("\nTransformer")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(10,5))
plt.plot(real,label="Real")
plt.plot(pred,label="Predicho")
plt.xlabel("Ventanas")
plt.ylabel("Generación (MWh)")
plt.title("Transformer - Predicción vs Real")
plt.legend()
plt.tight_layout()
plt.savefig(
    "prediccion_vs_real_transformer.png",
    dpi=300
)
plt.show()
plt.close()

archivo = "resultados.txt"

modo = "a" if os.path.exists(archivo) else "w"

with open(archivo, modo) as f:
    f.write("\n")
    f.write("MODELO: TRANSFORMER\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"])
plt.title("Curva de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "loss_transformer.png",
    dpi=300
)
plt.show()
plt.close()