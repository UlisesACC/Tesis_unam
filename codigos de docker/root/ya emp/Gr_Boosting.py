import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
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

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nGradient Boosting")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(12,5))
plt.plot(y_test.values,label="Real")
plt.plot(pred,label="Predicho")
plt.xlabel("Ventanas")
plt.ylabel("Generación (MWh)")
plt.title("Gradient Boosting")
plt.legend()
plt.tight_layout()
plt.savefig("prediccion_vs_real_gradient_boosting.png",dpi=300)
plt.show()
plt.close()