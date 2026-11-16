"""Compara Ridge, SVR y Transformer a una hora con rezagos comunes."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


HISTORIA = 24
ENTRENAMIENTO = 24
HORA = pd.Timedelta(hours=1)
OBJETIVO = "Generacion (MWh)"


def cargar_ejemplos(ruta):
    datos = pd.read_csv(ruta, usecols=["Area", "Timestamp", OBJETIVO])
    datos = datos.loc[datos["Area"].eq("BCA")].copy()
    datos["fecha"] = pd.to_datetime(datos["Timestamp"], format="%d/%m/%Y %H:%M")
    datos = datos.loc[datos["fecha"].dt.year.eq(2022)].sort_values("fecha")
    datos = datos.iloc[:int(0.7 * len(datos))]
    if datos["fecha"].duplicated().any():
        raise ValueError("Hay marcas temporales duplicadas en la serie de BCA.")
    generacion = pd.to_numeric(datos[OBJETIVO]).to_numpy(dtype=float)
    if len(generacion) <= HISTORIA or not np.isfinite(generacion).all():
        raise ValueError("La serie es insuficiente o contiene valores no finitos.")

    ventanas = np.lib.stride_tricks.sliding_window_view(generacion, HISTORIA + 1)
    # Cada ejemplo requiere continuidad tanto en los rezagos como en la respuesta.
    saltos = datos["fecha"].diff().iloc[1:].eq(HORA).to_numpy()
    validas = np.lib.stride_tricks.sliding_window_view(saltos, HISTORIA).all(axis=1)
    return ventanas[validas, :-1], ventanas[validas, -1], datos["fecha"].iloc[HISTORIA:].to_numpy()[validas]


def construir_transformer():
    capas = tf.keras.layers
    entrada = capas.Input(shape=(HISTORIA, 2))
    proyeccion = capas.Dense(16)(entrada)
    atencion = capas.MultiHeadAttention(num_heads=2, key_dim=16)(proyeccion, proyeccion)
    secuencia = capas.LayerNormalization()(proyeccion + capas.Dropout(0.1)(atencion))
    densa = capas.Dense(64, activation="relu")(secuencia)
    secuencia = capas.LayerNormalization()(secuencia + capas.Dense(16)(densa))
    salida = capas.Dense(1)(capas.GlobalAveragePooling1D()(secuencia))
    modelo = tf.keras.Model(entrada, salida)
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def como_secuencia(rezagos):
    posiciones = np.broadcast_to(np.linspace(-1, 1, HISTORIA), rezagos.shape)
    return np.stack((rezagos, posiciones), axis=-1).astype(np.float32)


def comparar(entradas, respuestas, fechas, pruebas, epocas):
    primer_origen = max(ENTRENAMIENTO, len(respuestas) - pruebas)
    filas = []
    for i in range(primer_origen, len(respuestas)):
        anteriores = slice(i - ENTRENAMIENTO, i)
        escalador_x = StandardScaler()
        x_train = escalador_x.fit_transform(entradas[anteriores])
        x_test = escalador_x.transform(entradas[i:i + 1])
        escalador_y = StandardScaler()
        y_train = escalador_y.fit_transform(respuestas[anteriores, None]).ravel()

        predicciones = {}
        for nombre, modelo in {
            "Ridge": Ridge(alpha=1.0),
            "SVR": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
        }.items():
            modelo.fit(x_train, y_train)
            predicciones[nombre] = modelo.predict(x_test)[0]

        # Se construye desde cero para cada origen, igual que los otros regresores.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        transformer = construir_transformer()
        transformer.fit(como_secuencia(x_train), y_train[:, None], epochs=epocas,
                        batch_size=8, shuffle=False, verbose=0)
        predicciones["Transformer"] = float(
            transformer(como_secuencia(x_test), training=False).numpy()[0, 0]
        )

        fila = {"fecha_hora": fechas[i], "real": respuestas[i]}
        for nombre, valor in predicciones.items():
            fila[nombre] = escalador_y.inverse_transform([[valor]])[0, 0]
        filas.append(fila)

    return pd.DataFrame(filas)


def guardar_resultados(predicciones, salida):
    salida.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(salida / "predicciones.csv", index=False)
    real = predicciones["real"]
    metricas = []
    for nombre in ("Ridge", "SVR", "Transformer"):
        estimado = predicciones[nombre]
        metricas.append({
            "modelo": nombre,
            "n_predicciones": len(real),
            "MAE_MWh": mean_absolute_error(real, estimado),
            "RMSE_MWh": np.sqrt(mean_squared_error(real, estimado)),
            "R2": r2_score(real, estimado) if real.nunique() > 1 else np.nan,
        })
    resumen = pd.DataFrame(metricas)
    resumen.to_csv(salida / "metricas.csv", index=False)
    print(resumen.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))

    grafica = predicciones.set_index("fecha_hora")
    grafica = grafica.loc[grafica.index < grafica.index[0] + pd.Timedelta(days=2)]
    ax = grafica.plot(figsize=(10, 4))
    ax.set(xlabel="Fecha y hora", ylabel="Generación (MWh)",
           title="Baja California, pronóstico a una hora")
    ax.figure.tight_layout()
    ax.figure.savefig(salida / "comparacion.png", dpi=180)
    plt.close(ax.figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path(__file__).with_name("datos.csv"))
    parser.add_argument("--max-pruebas", type=int, default=168)
    parser.add_argument("--epocas", type=int, default=50)
    parser.add_argument("--salida", type=Path, default=Path("comparacion_ridge_transformer_svm"))
    args = parser.parse_args()
    if args.max_pruebas < 2 or args.epocas < 1:
        parser.error("Se requieren al menos dos pruebas y una época.")
    entradas, respuestas, fechas = cargar_ejemplos(args.datos)
    if len(respuestas) < ENTRENAMIENTO + 2:
        parser.error("No hay suficientes ejemplos horarios válidos para entrenar y evaluar.")
    predicciones = comparar(entradas, respuestas, fechas, args.max_pruebas, args.epocas)
    print(f"Periodo evaluado: {predicciones['fecha_hora'].iloc[0]} a "
          f"{predicciones['fecha_hora'].iloc[-1]}")
    guardar_resultados(predicciones, args.salida)


if __name__ == "__main__":
    main()
