import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Считывание данных из файла
data = pd.read_csv('cars_data.csv', encoding='windows-1251', sep=';')

# Проверяем необходимые столбцы
required_cols = ['Цена', 'Класс', 'Пробег', 'Страна производства', 'Купили']
if not all(col in data.columns for col in required_cols):
    raise ValueError(f"В данных должны быть столбцы: {required_cols}")

print(data.head(5))

X = data[['Цена', 'Класс', 'Пробег', 'Страна производства']]
y = data['Купили'].astype(float)  # Целевая переменная бинарная, но для регрессии используем float

# Определяем числовые и категориальные признаки
numeric_features = ['Цена', 'Пробег']
categorical_features = ['Класс', 'Страна производства']

# Предобработка: масштабирование числовых и one-hot кодирование категориальных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=27)

def plot_regression_results(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Предсказанные значения')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Идеальная линия')
    for i in range(len(y_test)):
        plt.plot([y_test.iloc[i], y_test.iloc[i]], [y_pred[i], y_test.iloc[i]], color='gray', linestyle=':', linewidth=1)
    plt.title(f'Результаты регрессии: {model_name}')
    plt.xlabel('Реальные значения (Купили)')
    plt.ylabel('Предсказанные значения (вероятность)')
    plt.legend()
    plt.grid()
    plt.show()

# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression R²:", r2_score(y_test, y_pred_linear))
plot_regression_results(y_test, y_pred_linear, 'Линейная регрессия')

# Полиномиальная регрессия
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
X_poly_test = poly.transform(X_test)
y_pred_poly = poly_model.predict(X_poly_test)
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression R²:", r2_score(y_test, y_pred_poly))
plot_regression_results(y_test, y_pred_poly, 'Полиномиальная регрессия')

# Дерево решений
tree_model = DecisionTreeRegressor(random_state=27)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
print("Decision Tree Regression MSE:", mean_squared_error(y_test, y_pred_tree))
print("Decision Tree Regression R²:", r2_score(y_test, y_pred_tree))
plot_regression_results(y_test, y_pred_tree, 'Дерево решений')

# Регрессия LASSO
lasso_model = LassoCV(random_state=27)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print("Lasso Regression MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Lasso Regression R²:", r2_score(y_test, y_pred_lasso))
plot_regression_results(y_test, y_pred_lasso, 'Регрессия LASSO')

# Гребневая регрессия
ridge_model = RidgeCV()
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge Regression MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Ridge Regression R²:", r2_score(y_test, y_pred_ridge))
plot_regression_results(y_test, y_pred_ridge, 'Гребневая регрессия')

# Регрессия ElasticNet
elastic_model = ElasticNetCV(random_state=27)
elastic_model.fit(X_train, y_train)
y_pred_elastic = elastic_model.predict(X_test)
print("ElasticNet Regression MSE:", mean_squared_error(y_test, y_pred_elastic))
print("ElasticNet Regression R²:", r2_score(y_test, y_pred_elastic))
plot_regression_results(y_test, y_pred_elastic, 'Регрессия ElasticNet')