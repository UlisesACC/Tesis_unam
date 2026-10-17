import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

data = pd.read_csv("datos_ventanas.csv")

if "Timestamp_Target" in data.columns:
    data = data.drop(columns=["Timestamp_Target"])

X = data.drop(columns=["Target"])
y = data["Target"]

n_train = int(len(data) * 0.70)

X_train = X.iloc[:n_train]
X_test = X.iloc[n_train:]

y_train = y.iloc[:n_train]
y_test = y.iloc[n_train:]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nRandom Forest")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Real")
plt.plot(pred, label="Predicho")
plt.xlabel("Ventanas")
plt.ylabel("Generación (MWh)")
plt.title("Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("prediccion_vs_real_random_forest.png", dpi=300)
plt.show()
plt.close()