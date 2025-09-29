import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('cars_data.csv', encoding='windows-1251', sep=';')

# Предобработка признаков
# Выбираем нужные признаки и целевую переменную
features = data[['Цена', 'Класс', 'Пробег', 'Страна-производитель']]
target = data['Купили'].astype(int)  # Целевая переменная 0/1

# Кодируем категориальные признаки с помощью one-hot encoding
features_encoded = pd.get_dummies(features, columns=['Класс', 'Страна-производитель'], drop_first=True)

# Шаг 3: Нормализация числовых признаков
num_cols = ['Цена', 'Пробег']
scaler = StandardScaler()
features_encoded[num_cols] = scaler.fit_transform(features_encoded[num_cols])

# Шаг 4: PCA — определение количества компонент
pca = PCA()
pca.fit(features_encoded)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

print("Дисперсия для каждой главной компоненты:", explained_variance)
print("Кумулятивная дисперсия:", cumulative_variance)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
plt.title('Кумулятивная дисперсия главных компонент')
plt.xlabel('Компонента')
plt.ylabel('Кумулятивная дисперсия')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=2, color='g', linestyle='--')
plt.grid()
plt.show()

# Шаг 5: Снижение размерности до 2 компонент
pca = PCA(n_components=2)
X_pca_reduced = pca.fit_transform(features_encoded)
print("Доля дисперсии для 2 компонент:", pca.explained_variance_ratio_)

# Шаг 6: Визуализация результатов
pca_df = pd.DataFrame(X_pca_reduced, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Купили'] = target.astype(str)
pca_df['Купили'] = pca_df['Купили'].map({'0': 'Нет', '1': 'Да'})

plt.figure(figsize=(10, 10))
sns.scatterplot(data=pca_df, x='Principal Component 1', y='Principal Component 2',
                hue='Купили', palette={'Нет': 'red', 'Да': 'blue'}, alpha=0.7)
plt.title('PCA: 2 главные компоненты')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Купили')
plt.grid()
plt.show()