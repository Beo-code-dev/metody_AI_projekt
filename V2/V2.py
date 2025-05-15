import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

steam_games = pd.read_csv("games_march2025_cleaned.csv")
df = steam_games.copy()

# Ograniczenie do gier płatnych
df = df[df['price'] > 0]

# Przykład ekstrakcji cech tekstowych (liczba tagów)
df['tag_count'] = df['tags'].apply(lambda x: len(str(x).split(',')))

# Cechy do modelowania
features = ['required_age', 'dlc_count', 'average_playtime_2weeks', 'tag_count', 'discount']
target = 'price'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Regresja Liniowa"""

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

"""Random Forest"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

"""Ocena modeli"""

def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Metrics:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.2f}")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

"""Zapis Modeli w folderze models (oraz stworzenie tego folderu, jeśli nie istnieje)"""

os.makedirs('models', exist_ok=True)
joblib.dump(lr_model, 'models/model_lr_v2.pkl')
joblib.dump(rf_model, 'models/model_rf_v2.pkl')

"""Wizualizacja"""

errors_rf = y_pred_rf - y_test
errors_lr = y_pred_lr - y_test

plt.figure(figsize=(10, 5))
plt.hist(errors_rf, bins=50, alpha=0.5, label="Random Forest")
plt.hist(errors_lr, bins=50, alpha=0.5, label="Linear Regression")
plt.xlim(-100, 100)  # ograniczenie widocznych błędów do typowego zakresu
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Games")
plt.title("Histogram of Prediction Errors")
plt.legend()
plt.grid(True)
plt.show()