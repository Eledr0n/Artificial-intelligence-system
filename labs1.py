import statistics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# 2. Загрузить датасет в pandas DataFrame и вывести на экран
df = pd.read_csv("cars_data.csv", encoding='windows-1251', sep=';')
print(df.head())

# 3. Кодирование категориальных признаков 'Класс' и 'Страна производства'
columns_to_encode = ['Класс', 'Страна производства']
encoder = OneHotEncoder(sparse_output=False, dtype=np.float64)
encoded_data = encoder.fit_transform(df[columns_to_encode])

# Создаем имена новых столбцов
new_subcolumns_names = encoder.get_feature_names_out(columns_to_encode)
encoded_df = pd.DataFrame(encoded_data, columns=new_subcolumns_names, index=df.index)

# Удаляем исходные столбцы
df.drop(columns_to_encode, axis=1, inplace=True)

# Добавляем новые закодированные столбцы
df = pd.concat([df, encoded_df], axis=1)
print("\nДанные после кодирования:")
print(df.head())

# Преобразуем целевой признак 'Купили' в числовой (если нужно)
if df['Купили'].dtype == object:
    df['Купили'] = df['Купили'].map({'нет': 0, 'да': 1})

# 4. Определить среднее значение, медиану, моду и стандартное отклонение для числовых признаков
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

print("\n....Ковариация----------------------------")
print(cov)
print("\n....Корреляция----------------------------")
print(cor)

# 6. Визуализация данных
sns.pairplot(df, vars=numeric_cols, hue='Купили')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cor, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Тепловая карта корреляции')
plt.show()

# 7. Нормализация числовых признаков
scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
print('\n----------------------------Нормированные данные----------------------------')
print(scaled_df.head())