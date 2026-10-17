import pandas as pd

# ==========================
# Configuración
# ==========================
ARCHIVO_ENTRADA = "datos.csv"
ARCHIVO_SALIDA = "datos_ventanas.csv"

WINDOW_SIZE = 25

TARGET = "Generacion (MWh)"

FEATURES = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday"
]

# ==========================
# Leer datos
# ==========================
data = pd.read_csv(ARCHIVO_ENTRADA)

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

data["Hour_of_Day"] = data["Hour"] + data["Minute"]/60
data["Day_of_Year"] = data["fecha_hora"].dt.dayofyear
data["Weekday"] = data["fecha_hora"].dt.weekday

# ==========================
# Filtrado (igual que tu código)
# ==========================
fecha_inicio = "2022-01-01"
fecha_fin = "2022-12-31"

area = data["Area"].unique()[0]

data = data[
    (data["fecha_hora"] >= fecha_inicio) &
    (data["fecha_hora"] <= fecha_fin) &
    (data["Area"] == area)
].reset_index(drop=True)

# ==========================
# Construcción de ventanas
# ==========================
filas = []

for i in range(len(data) - WINDOW_SIZE + 1):

    ventana = data.iloc[i:i+WINDOW_SIZE]

    fila = {}

    # Primeros 24 registros
    for j in range(WINDOW_SIZE - 1):

        for f in FEATURES:
            fila[f"{f}_t{j+1}"] = ventana.iloc[j][f]

        fila[f"{TARGET}_t{j+1}"] = ventana.iloc[j][TARGET]

    # Objetivo (última fila)
    fila["Target"] = ventana.iloc[-1][TARGET]

    # Fecha del objetivo
    fila["Timestamp_Target"] = ventana.iloc[-1]["Timestamp"]

    filas.append(fila)

# ==========================
# Guardar
# ==========================
df = pd.DataFrame(filas)

# Dejar Timestamp al inicio
cols = ["Timestamp_Target"] + [c for c in df.columns if c != "Timestamp_Target"]
df = df[cols]

df.to_csv(ARCHIVO_SALIDA, index=False)

print("--------------------------------")
print("Ventanas creadas correctamente")
print("Archivo:", ARCHIVO_SALIDA)
print("Número de ventanas:", len(df))
print("Número de columnas:", len(df.columns))
print("--------------------------------")