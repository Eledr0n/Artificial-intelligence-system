import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN

# Чтение данных из файла
data = pd.read_csv('cars_data.csv', encoding='windows-1251', sep=';')

print(data.head())

# Выбираем признаки и целевую переменную
X = data[['Цена', 'Класс', 'Пробег', 'Страна производства']]
y = data['Купили']

# Предобработка: масштабируем числовые признаки и кодируем категориальные
numeric_features = ['Цена', 'Пробег']
categorical_features = ['Класс', 'Страна производства']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# Метод локтя для выбора оптимального количества кластеров
inertia = []
k_values = range(1, 11)  # Проверяем k от 1 до 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)

# Визуализация метода локтя
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Метод локтя для выбора оптимального количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Сумма квадратов расстояний (Inertia)')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# ===Кластеризация методом k-средних===
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_processed)
kmeans_labels = kmeans.labels_

# Оценка качества кластеризации (класс — бинарный, но используем для оценки)
ari_kmeans = metrics.adjusted_rand_score(y, kmeans_labels)
ami_kmeans = metrics.adjusted_mutual_info_score(y, kmeans_labels)
homogeneity_kmeans = metrics.homogeneity_score(y, kmeans_labels)
completeness_kmeans = metrics.completeness_score(y, kmeans_labels)
v_measure_kmeans = metrics.v_measure_score(y, kmeans_labels)
silhouette_kmeans = metrics.silhouette_score(X_processed, kmeans_labels)
print(f'K-средние:\nARI: {ari_kmeans}, AMI: {ami_kmeans}, Homogeneity: {homogeneity_kmeans}, Completeness: {completeness_kmeans}, V-measure: {v_measure_kmeans}, Silhouette: {silhouette_kmeans}')
print('Номера кластеров (K-средние):', kmeans_labels)

# Для визуализации выберем первые два числовых признака после масштабирования: 'Цена' и 'Пробег'
price_idx = 0
mileage_idx = 1

plt.figure(figsize=(8, 6))
plt.scatter(X_processed[:, price_idx], X_processed[:, mileage_idx], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-средние: Кластеры')
plt.xlabel('Цена (масштабированная)')
plt.ylabel('Пробег (масштабированный)')
plt.colorbar(label='Номера кластеров')
plt.grid()
plt.show()

# ===Иерархическая кластеризация===
mergings = linkage(X_processed, method='ward')
hierarchical_labels = fcluster(mergings, t=k, criterion='maxclust')

ari_hierarchical = metrics.adjusted_rand_score(y, hierarchical_labels)
ami_hierarchical = metrics.adjusted_mutual_info_score(y, hierarchical_labels)
homogeneity_hierarchical = metrics.homogeneity_score(y, hierarchical_labels)
completeness_hierarchical = metrics.completeness_score(y, hierarchical_labels)
v_measure_hierarchical = metrics.v_measure_score(y, hierarchical_labels)
silhouette_hierarchical = metrics.silhouette_score(X_processed, hierarchical_labels) if len(set(hierarchical_labels)) > 1 else -1
print(f'Иерархическая кластеризация:\nARI: {ari_hierarchical}, AMI: {ami_hierarchical}, Homogeneity: {homogeneity_hierarchical}, Completeness: {completeness_hierarchical}, V-measure: {v_measure_hierarchical}, Silhouette: {silhouette_hierarchical}')
print('Номера кластеров (Иерархическая кластеризация):', hierarchical_labels)

plt.figure(figsize=(10, 7))
dendrogram(mergings, labels=y.values, leaf_rotation=90, leaf_font_size=12)
plt.title('Дендрограмма иерархической кластеризации')
plt.xlabel('Объекты')
plt.ylabel('Расстояние')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_processed[:, price_idx], X_processed[:, mileage_idx], c=hierarchical_labels, cmap='viridis', marker='o')
plt.title('Иерархическая кластеризация: Кластеры')
plt.xlabel('Цена (масштабированная)')
plt.ylabel('Пробег (масштабированный)')
plt.colorbar(label='Номера кластеров')
plt.grid()
plt.show()

# ===Кластеризация на основе плотности (DBSCAN)===
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_processed)

dbscan_ari = metrics.adjusted_rand_score(y, dbscan_labels)
dbscan_ami = metrics.adjusted_mutual_info_score(y, dbscan_labels)
dbscan_homogeneity = metrics.homogeneity_score(y, dbscan_labels)
dbscan_completeness = metrics.completeness_score(y, dbscan_labels)
dbscan_v_measure = metrics.v_measure_score(y, dbscan_labels)
silhouette_dbscan = metrics.silhouette_score(X_processed, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
print(f'DBSCAN:\nARI: {dbscan_ari}, AMI: {dbscan_ami}, Homogeneity: {dbscan_homogeneity}, Completeness: {dbscan_completeness}, V-measure: {dbscan_v_measure}, Silhouette: {silhouette_dbscan}')
print('Номера кластеров (DBSCAN):', dbscan_labels)

plt.figure(figsize=(8, 6))
plt.scatter(X_processed[:, price_idx], X_processed[:, mileage_idx], c=dbscan_labels, cmap='viridis', marker='o')
plt.title('DBSCAN: Кластеры')
plt.xlabel('Цена (масштабированная)')
plt.ylabel('Пробег (масштабированный)')
plt.colorbar(label='Номера кластеров (шум -1)')
plt.grid()
plt.show()