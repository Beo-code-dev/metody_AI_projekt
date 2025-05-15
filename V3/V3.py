import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

steam_games = pd.read_csv("games_march2025_cleaned.csv")
df = steam_games.copy()

# Przekształcenie daty wydania na typ datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Filtr gier na trzy lata wstecz
cutoff_date = pd.Timestamp('2022-03-01')
df = df[df['release_date'] >= cutoff_date]

# Filtr: tylko gry strategiczne (gatunek zawiera "Strategy")
df = df[df['genres'].str.contains('Strategy', na=False)]

# Ograniczenie do gier płatnych
df = df[df['price'] > 0]

# Ekstrakcja cech tekstowych
df['tag_count'] = df['tags'].apply(lambda x: len(str(x).split(',')))

# Cechy do modelowania
features = ['required_age', 'dlc_count', 'average_playtime_2weeks', 'tag_count', 'discount']
target = 'price'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Optymalizacja Regresji"""

param_grid_ridge = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search_ridge = GridSearchCV(Ridge(), param_grid_ridge, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_ridge.fit(X_train, y_train)
best_ridge = grid_search_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

"""Optymalizacja Random Forest"""

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

"""Ocena"""

def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Metrics:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.2f}")

evaluate(y_test, y_pred_ridge, "Optimized Linear Regression")
evaluate(y_test, y_pred_rf, "Optimized Random Forest")

os.makedirs('models', exist_ok=True)
joblib.dump(best_ridge, 'models/model_lr_v3.pkl')
joblib.dump(best_rf, 'models/model_rf_v3.pkl')

"""Wizualizacja"""

errors_ridge = y_pred_ridge - y_test
errors_rf = y_pred_rf - y_test

plt.figure(figsize=(10, 5))
plt.hist(errors_ridge, bins=50, alpha=0.5, label="Ridge Regression")
plt.hist(errors_rf, bins=50, alpha=0.5, label="Random Forest")
plt.xlim(-100, 100)
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Games")
plt.title("Histogram of Prediction Errors")
plt.legend()
plt.grid(True)
plt.show()