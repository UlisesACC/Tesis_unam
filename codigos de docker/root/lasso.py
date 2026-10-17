#importando las librerias
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

matplotlib.use("Agg")

data=pd.read_csv("datos.csv")
time = data["Timestamp"]
time = pd.to_datetime(time, format="%d/%m/%Y %H:%M", dayfirst=True)
#obteniendo los anios
years = time.dt.year.unique()
#agregando una columna con el anio
data["Year"] = time.dt.year
#obteniendo los meses
months = time.dt.month.unique()
#agregando una columna con el mes
data["Month"] = time.dt.month
#obteniendo los dias
days = time.dt.day.unique()
#agregando una columna con el dia
data["Day"] = time.dt.day
#obteniendo las horas
hours = time.dt.hour.unique()
#agregando una columna con la hora
data["Hour"] = time.dt.hour
#obteniendo los minutos
minutes = time.dt.minute.unique()
#agregando una columna con el minuto
data["Minute"] = time.dt.minute
#obteniendo los segundos
seconds = time.dt.second.unique()
#agregando una columna con el segundo
data["Second"] = time.dt.second
#obteniendo los microsegundos
microseconds = time.dt.microsecond.unique()
#agregando una columna con el microsegundo
data["Microsecond"] = time.dt.microsecond
#eliminando la columna Timestamp
#data.drop(columns=["Timestamp"], inplace=True)

data["fecha_hora"] = pd.to_datetime(data[["Year", "Month", "Day", "Hour"]])
agrupado = data.groupby(["fecha_hora", "Area"])["Generacion (MWh)"].sum().reset_index()
# dividiendo por areas
areas = agrupado["Area"].unique()
num_areas = len(areas)

data["fecha_hora"] = pd.to_datetime(data[["Year", "Month", "Day", "Hour"]])
agrupado = data.groupby(["fecha_hora", "Area"])["Generacion (MWh)"].sum().reset_index()



warnings.filterwarnings("ignore")

data['Hour_of_Day'] = data['Hour'] + data['Minute'] / 60.0
data['Day_of_Year'] = data['fecha_hora'].dt.dayofyear
data['Year'] = data['fecha_hora'].dt.year
data['Month'] = data['fecha_hora'].dt.month
data['Weekday'] = data['fecha_hora'].dt.weekday

fecha_inicio = '2022-01-01'
fecha_fin = '2022-12-31'
area_seleccionada = data['Area'].unique()[0]

data_periodo = data[
    (data['fecha_hora'] >= fecha_inicio) &
    (data['fecha_hora'] <= fecha_fin) &
    (data['Area'] == area_seleccionada)
].copy()

target = 'Generacion (MWh)'

data_periodo['Gen_lag1'] = data_periodo[target].shift(1)

data_periodo_lag = data_periodo.dropna(subset=['Gen_lag1']).copy()

features = ['Hour_of_Day', 'Day_of_Year', 'Year', 'Month', 'Weekday', 'Gen_lag1']

n_total = len(data_periodo_lag)
n_70 = int(n_total * 0.7)

data_70 = data_periodo_lag.iloc[:n_70].copy()

window_size = 13

reales = []
predichos = []

for i in range(len(data_70) - window_size + 1):
    ventana = data_70.iloc[i:i + window_size]
    train_12 = ventana.iloc[:-1]
    test_1 = ventana.iloc[-1:]

    X_train = train_12[features]
    y_train = train_12[target]

    X_test = test_1[features]
    y_test = test_1[target]

    model = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=3)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)[0]

    reales.append(y_test.values[0])
    predichos.append(y_test_pred)

reales = np.array(reales)
predichos = np.array(predichos)

mae = mean_absolute_error(reales, predichos)
rmse = np.sqrt(mean_squared_error(reales, predichos))
r2 = r2_score(reales, predichos)

plt.figure(figsize=(8, 5))
plt.plot(reales, color='blue', label='Real')
plt.plot(predichos, color='orange', label='Predicho')
plt.xlabel('Horas')
plt.ylabel('(MWh)')
plt.title('Predicción vs Real')
plt.legend()
plt.show()
plt.savefig("prediccion_vs_real_lasso.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
