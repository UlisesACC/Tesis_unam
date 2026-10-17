import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

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

n_70 = int(len(data_periodo) * 0.7)
data_70 = data_periodo.iloc[:n_70].copy()

features = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday"
]

def build_transformer(input_shape):
    inputs = Input(shape=input_shape)

    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
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


window_size = 25
reales = []
predichos = []

for i in range(len(data_70) - window_size + 1):
    ventana = data_70.iloc[i:i + window_size]

    train = ventana.iloc[:-1]
    test = ventana.iloc[-1:]

    X_train = train[features].values
    y_train = train[target].values

    X_test = test[features].values
    y_test = test[target].values

    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = build_transformer((1, len(features)))

    model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=50,
        batch_size=8,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test_scaled, verbose=0)[0][0]
    y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]

    reales.append(y_test[0])
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
plt.title("Transformer + MinMaxScaler – Predicción vs Real")
plt.legend()
plt.savefig("prediccion_vs_real_transformer.png", dpi=300, bbox_inches="tight")
plt.close()

print("MÉTRICAS GLOBALES (TRANSFORMER)")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")