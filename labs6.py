import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns


#создадим синтетические данные:
np.random.seed(42)
n_samples = 500
price = np.random.uniform(5000, 50000, n_samples)
car_class = np.random.choice(['A', 'B', 'C'], n_samples)
mileage = np.random.uniform(0, 200000, n_samples)
country = np.random.choice(['Германия', 'Япония', 'Россия'], n_samples)

# Целевая переменная — вероятность продажи (0 или 1)
# Для примера создадим зависимость от цены и пробега
buy_prob = 1 / (1 + np.exp((price - 20000)/5000 + (mileage - 50000)/20000))
bought = np.random.binomial(1, buy_prob)

data = pd.DataFrame({
    'Цена': price,
    'Класс': car_class,
    'Пробег': mileage,
    'Страна производства': country,
    'Купили': bought
})

print(data.head())

# --- Предобработка ---

# Масштабируем числовые признаки
numeric_features = ['Цена', 'Пробег']
scaler = MinMaxScaler()
X_num = scaler.fit_transform(data[numeric_features])

# Кодируем категориальные признаки
categorical_features = ['Класс', 'Страна производства']
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_cat = encoder.fit_transform(data[categorical_features])

# Объединяем признаки
X = np.hstack([X_num, X_cat])
y = data['Купили'].values.reshape(-1, 1).astype(np.float32)

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Преобразуем в тензоры
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# --- Определение модели ---

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SimpleNN(X.shape[1])

criterion = nn.BCELoss()  # Для вероятности продажи (бинарная классификация)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Обучение модели ---

epochs = 500
batch_size = 64
losses = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])

    epoch_loss = 0
    for i in range(0, X_train_tensor.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    epoch_loss /= X_train_tensor.size()[0]
    losses.append(epoch_loss)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Эпоха {epoch+1}/{epochs}, Потеря: {epoch_loss:.4f}")

# --- Оценка модели ---

model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy()
    y_test_pred = model(X_test_tensor).numpy()

train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)
print(f"ROC AUC на обучающей выборке: {train_auc:.3f}")
print(f"ROC AUC на тестовой выборке: {test_auc:.3f}")

# --- Визуализация ---

# 1. График потерь обучения
plt.figure(figsize=(8,5))
plt.plot(losses, label='Потеря на обучении')
plt.xlabel('Эпоха')
plt.ylabel('Потеря (BCELoss)')
plt.title('График потерь при обучении')
plt.legend()
plt.grid()
plt.show()

# 2. График распределения вероятностей предсказания для классов
plt.figure(figsize=(8,5))
plt.hist(y_test_pred[y_test.flatten()==0], bins=30, alpha=0.6, label='Класс 0 (не купили)')
plt.hist(y_test_pred[y_test.flatten()==1], bins=30, alpha=0.6, label='Класс 1 (купили)')
plt.xlabel('Предсказанная вероятность покупки')
plt.ylabel('Количество')
plt.title('Распределение предсказанных вероятностей на тестовой выборке')
plt.legend()
plt.grid()
plt.show()

# 3. Зависимость вероятности покупки от цены (усреднённая по классам и странам)
data['Вероятность покупки'] = model(torch.from_numpy(X).float()).detach().numpy()
plt.figure(figsize=(8,5))
plt.scatter(data['Цена'], data['Вероятность покупки'], alpha=0.5)
plt.xlabel('Цена')
plt.ylabel('Предсказанная вероятность покупки')
plt.title('Зависимость вероятности покупки от цены')
plt.grid()
plt.show()

# 4. Boxplot вероятностей покупки по классам автомобиля
plt.figure(figsize=(8,5))
sns.boxplot(x='Класс', y='Вероятность покупки', data=data)
plt.title('Распределение вероятности покупки по классам автомобиля')
plt.grid()
plt.show()
