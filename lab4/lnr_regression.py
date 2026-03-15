import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

df.replace(-200, np.nan, inplace=True)
df = df.fillna(df.mean(numeric_only=True))

y = df['CO(GT)']

# 1
X1 = df[['C6H6(GT)']]

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== Задание 1 ===")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"Коэффициент: {model.coef_[0]:.4f}, intercept: {model.intercept_:.4f}")

# 2
features2 = ['C6H6(GT)', 'T', 'RH', 'NO2(GT)']
X2 = df[features2]

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== Задание 2 ===")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print("Коэффициенты:", dict(zip(features2, model.coef_)))

# 3 (те же признаки, что в 2)
features3 = ['C6H6(GT)', 'T', 'RH', 'NO2(GT)']
X3 = df[features3]

X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== Задание 3 ===")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print("Коэффициенты (после стандартизации):", dict(zip(features3, model.coef_)))

# 4 (все числовые признаки)
all_features = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
                'PT08.S5(O3)', 'T', 'RH', 'AH']
X4 = df[all_features]

X_train, X_test, y_train, y_test = train_test_split(X4, y, test_size=0.2, random_state=42)

param_grid = {'alpha': np.logspace(-4, 1, 20)}
grid = GridSearchCV(Lasso(max_iter=10000), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

model = grid.best_estimator_
best_alpha = grid.best_params_['alpha']
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
nonzero = np.sum(model.coef_ != 0)

print("=== Задание 4 ===")
print(f"Оптимальное alpha = {best_alpha:.6f}")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
print(f"Ненулевых коэффициентов: {nonzero} из {len(all_features)}")

# 5
X5 = df[all_features]

X_train, X_test, y_train, y_test = train_test_split(X5, y, test_size=0.2, random_state=42)

param_grid = {'alpha': np.logspace(-4, 1, 20)}
grid = GridSearchCV(Ridge(max_iter=10000), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

model = grid.best_estimator_
best_alpha = grid.best_params_['alpha']
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== Задание 5 ===")
print(f"Оптимальное alpha = {best_alpha:.6f}")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")

# 6 (признаки из 2)
X6 = df[['C6H6(GT)', 'T', 'RH', 'NO2(GT)']]

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(model, X6, y, cv=kf, scoring='r2')
mse_scores = -cross_val_score(model, X6, y, cv=kf, scoring='neg_mean_squared_error')

print("=== Задание 6 ===")
print("R² по фолдам:", np.round(r2_scores, 4))
print(f"Средний R² = {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
print("MSE по фолдам:", np.round(mse_scores, 4))
print(f"Средний MSE = {mse_scores.mean():.4f}")

# 7 (пример с RidgeCV)
X7 = df[all_features]

X_train, X_test, y_train, y_test = train_test_split(X7, y, test_size=0.2, random_state=42)

model = RidgeCV(alphas=np.logspace(-4, 1, 20), cv=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
best_alpha = model.alpha_

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== Задание 7 ===")
print(f"Оптимальное alpha (RidgeCV) = {best_alpha:.6f}")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")

# 8 (признаки из задания 2)
features8 = ['C6H6(GT)', 'T', 'RH', 'NO2(GT)']
X_lin = df[features8]   # для сравнения с линейной
X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_lin)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2_poly = r2_score(y_test, y_pred)
mse_poly = mean_squared_error(y_test, y_pred)

# Сравнение с обычной линейной (из задания 2)
print("=== Задание 8 ===")
print(f"Полиномиальная (degree=2) → R² = {r2_poly:.4f}, MSE = {mse_poly:.4f}")
print(f"Обычная линейная (задание 2) → R² ≈ 0.XX, MSE ≈ X.XX  (зависит от запуска)")

# 9 (собираем модели)
print("=== Задание 9: Сравнение моделей ===")

# 1. Простая линейная (C6H6)
X1 = df[['C6H6(GT)']]
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
m1: LinearRegression = LinearRegression().fit(X_train, y_train)
print(f"1. Простая LR      → R² = {r2_score(y_test, m1.predict(X_test)):.4f}")

# 2. Множественная LR
X2 = df[['C6H6(GT)', 'T', 'RH', 'NO2(GT)']]
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
m2 = LinearRegression().fit(X_train, y_train)
print(f"2. Множественная LR → R² = {r2_score(y_test, m2.predict(X_test)):.4f}")

# 3. Lasso (все признаки)
X_all = df[all_features]
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
lasso = Lasso(alpha=0.001, max_iter=10000).fit(X_train, y_train)  # пример alpha
print(f"4. Lasso            → R² = {r2_score(y_test, lasso.predict(X_test)):.4f}")

# 4. Ridge (все признаки)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
print(f"5. Ridge            → R² = {r2_score(y_test, ridge.predict(X_test)):.4f}")

# 5. Полиномиальная (degree=2)
poly_feat = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_feat.fit_transform(X2)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)
mp = LinearRegression().fit(X_train_p, y_train_p)
print(f"8. Полиномиальная   → R² = {r2_score(y_test_p, mp.predict(X_test_p)):.4f}")

# Интерпретация коэффициентов
print("\nИнтерпретация коэффициентов")
coef_dict = dict(zip(['C6H6(GT)', 'T', 'RH', 'NO2(GT)'], m2.coef_))
sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
for feat, val in sorted_coef:
    print(f"  {feat:12} : {val:8.4f}")
