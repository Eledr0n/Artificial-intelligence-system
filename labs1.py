import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 11)

# Загрузить датасет в DataFrame и вывести на экран
df = pd.read_csv("cars_data.csv", encoding='windows-1251', sep=';')
print(df.head())

# Кодирование категориальных признаков 'Класс' и 'Страна-производитель'
columns_to_encode = ['Класс', 'Страна-производитель']
encoder = OneHotEncoder(sparse_output=False, dtype=np.float64)
encoded_data = encoder.fit_transform(df[columns_to_encode])

# Создаем имена новых столбцов
new_subcolumns_names = encoder.get_feature_names_out(columns_to_encode)
encoded_df = pd.DataFrame(encoded_data, columns=new_subcolumns_names, index=df.index)

df.drop(columns_to_encode, axis=1, inplace=True)
df = pd.concat([df, encoded_df], axis=1)
print("\nДанные после кодирования:")
print(df.head())

# if df['Купили'].dtype == object:
#     df['Купили'] = df['Купили'].map({'нет': 0, 'да': 1})

# Среднее значение, медиана, мода и стандартное отклонение
numeric_cols = ['Цена', 'Пробег']

print(f"\n....Среднее значение----------------------------")
for col in numeric_cols:
    print(f'{col}: {statistics.mean(df[col])}')

print(f"\n....Медиана----------------------------")
for col in numeric_cols:
    print(f'{col}: {statistics.median(df[col])}')

print(f"\n....Мода----------------------------")
for col in numeric_cols:
    try:
        print(f'{col}: {statistics.mode(df[col])}')
    except statistics.StatisticsError:
        print(f'{col}: Мода не определена (несколько значений)')

print(f"\n....Стандартное отклонение----------------------------")
for col in numeric_cols:
    print(f'{col}: {statistics.stdev(df[col])}')

# 5. Определить корреляцию между числовыми признаками и целевым признаком
corr_cols = numeric_cols + ['Купили']
cov = df[corr_cols].cov()
cor = df[corr_cols].corr()

print("\n-----Ковариация----------------------------")
print(cov)
print("\n-----Корреляция----------------------------")
print(cor)

# 6. Визуализация
sns.pairplot(df, vars=numeric_cols, hue='Купили')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cor, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Тепловая карта корреляции')
plt.show()

# 7. Нормализация числовых признаков - ИСПРАВЛЕННАЯ ВЕРСИЯ
scaler = preprocessing.MinMaxScaler()

# Сохраняем нормализованные данные обратно в основной DataFrame
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print('\n----------------------------Нормированные данные----------------------------')
print(df[numeric_cols].head())