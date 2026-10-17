import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


data['Hour_of_Day'] = data['Hour'] + data['Minute'] / 60.0


data['Hour_sin'] = np.sin(2 * np.pi * data['Hour_of_Day'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour_of_Day'] / 24)


area_seleccionada = data['Area'].unique()[0]
data_area = data[data['Area'] == area_seleccionada].copy()

features = ['Hour_of_Day', 'Hour_sin', 'Hour_cos']
target = 'Generacion (MWh)'


ultimos_12 = data_area.tail(12)

X_train = ultimos_12[features]
y_train = ultimos_12[target]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


kf = KFold(
    n_splits=min(5, len(X_train)),
    shuffle=True,
    random_state=42
)


param_grid = {
    'hidden_layer_sizes': [(32,16), (64,32,16), (128,64,32)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
}

mlp = MLPRegressor(
    solver='adam',
    max_iter=10000,
    early_stopping=True,
    random_state=42,
    n_iter_no_change=100
)

grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='r2',
    cv=kf,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_scaled, y_train)
best_model = grid.best_estimator_

print("\n=== MEJOR MODELO MLP ===")
print(f"Capas ocultas: {best_model.hidden_layer_sizes}")
print(f"Activación: {best_model.activation}")
print(f"Alpha: {best_model.alpha}")
print(f"Learning rate: {best_model.learning_rate_init}")
print(f"R² CV: {grid.best_score_:.4f}")

y_pred = best_model.predict(X_scaled)

print("\n=== MÉTRICAS ENTRENAMIENTO ===")
print(f"MSE: {mean_squared_error(y_train, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred):.4f}")
print(f"R²: {r2_score(y_train, y_pred):.4f}")
print(f"Iteraciones: {best_model.n_iter_}")

plt.figure(figsize=(10,4))
plt.plot(best_model.loss_curve_)
plt.title("Curva de pérdida – MLP")
plt.xlabel("Iteraciones")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


last_hour = ultimos_12['Hour_of_Day'].iloc[-1]
future_hours = (last_hour + np.arange(1,13)) % 24

future_df = pd.DataFrame({
    'Hour_of_Day': future_hours
})

future_df['Hour_sin'] = np.sin(2 * np.pi * future_df['Hour_of_Day'] / 24)
future_df['Hour_cos'] = np.cos(2 * np.pi * future_df['Hour_of_Day'] / 24)

X_future = scaler.transform(future_df[features])
y_future = best_model.predict(X_future)

plt.figure(figsize=(12,5))
plt.plot(range(12), y_train.values, marker='o', label='Últimos reales')
plt.plot(range(12,24), y_future, marker='o', label='Predicción 12h')
plt.axvline(11, color='red', linestyle='--')
plt.title(f'MLP Multicapa – Área {area_seleccionada}')
plt.xlabel('Índice temporal')
plt.ylabel(target)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()