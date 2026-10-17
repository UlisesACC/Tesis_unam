import pandas as pd

WINDOW_SIZE = 24

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
    data[["Year","Month","Day","Hour"]]
)

data["Hour_of_Day"] = data["Hour"] + data["Minute"]/60
data["Day_of_Year"] = data["fecha_hora"].dt.dayofyear
data["Weekday"] = data["fecha_hora"].dt.weekday

fecha_inicio = "2022-01-01"
fecha_fin = "2022-12-31"

area = data["Area"].unique()[0]

data = data[
    (data["fecha_hora"] >= fecha_inicio) &
    (data["fecha_hora"] <= fecha_fin) &
    (data["Area"] == area)
].reset_index(drop=True)

features = [
    "Hour_of_Day",
    "Day_of_Year",
    "Year",
    "Month",
    "Weekday",
    "Generacion (MWh)"
]

rows = []

for i in range(len(data)-WINDOW_SIZE):

    row = {}

    row["Window_ID"] = i
    row["Start_Index"] = i
    row["Timestamp_Target"] = data.iloc[i+WINDOW_SIZE]["Timestamp"]

    for j in range(WINDOW_SIZE):

        for f in features:

            row[f"{f}_t{j+1}"] = data.iloc[i+j][f]

    row["Target"] = data.iloc[i+WINDOW_SIZE]["Generacion (MWh)"]

    rows.append(row)

pd.DataFrame(rows).to_csv(
    "datos_backtesting.csv",
    index=False
)

print("CSV generado")