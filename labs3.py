import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

# Загрузка данных из файла
data = pd.read_csv('cars_data.csv', encoding='windows-1251', sep=';')

# Проверка на наличие необходимых столбцов
required_cols = ['Цена', 'Класс', 'Пробег', 'Страна производства', 'Купили']
if not all(col in data.columns for col in required_cols):
    raise ValueError(f"Файл должен содержать столбцы: {required_cols}")

print("Проверка на NaN значения:\n", data.isnull().sum())

# Разделение данных на признаки и метки
features = data[['Цена', 'Класс', 'Пробег', 'Страна производства']]
labels = data['Купили']

# Обработка признаков: числовые - масштабирование, категориальные - OneHotEncoding
numeric_features = ['Цена', 'Пробег']
categorical_features = ['Класс', 'Страна производства']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(features)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_processed, labels, test_size=0.3, random_state=27)

# Функция оценки качества модели
def evaluate_model(y_true, y_pred):
    print("Матрица ошибок:\n", confusion_matrix(y_true, y_pred))
    print("Точность:", accuracy_score(y_true, y_pred))
    print("Отчет по классификации:\n", classification_report(y_true, y_pred))

# Визуализация матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Матрица ошибок: {model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Не купили', 'Купили'])
    plt.yticks(tick_marks, ['Не купили', 'Купили'])
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.show()

# Построение ROC-кривой
def plot_roc_curve(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        # Масштабируем в [0,1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Ложноположительные')
    plt.ylabel('Истинноположительные')
    plt.title(f'ROC кривая: {model_name}')
    plt.legend(loc='lower right')
    plt.show()


# Логистическая регрессия
logreg_clf = LogisticRegression(max_iter=1000)
logreg_clf.fit(X_train, y_train)
y_pred_logreg = logreg_clf.predict(X_test)

print("Логистическая регрессия:")
evaluate_model(y_test, y_pred_logreg)
plot_confusion_matrix(y_test, y_pred_logreg, "Логистическая регрессия")
plot_roc_curve(logreg_clf, X_test, y_test, "Логистическая регрессия")

# K-NN
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

print("\nK-NN:")
evaluate_model(y_test, y_pred_knn)
plot_confusion_matrix(y_test, y_pred_knn, "K-NN")
plot_roc_curve(knn_clf, X_test, y_test, "K-NN")

# Решающее дерево
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

print("\nРешающее дерево:")
evaluate_model(y_test, y_pred_tree)
plot_confusion_matrix(y_test, y_pred_tree, "Решающее дерево")
plot_roc_curve(tree_clf, X_test, y_test, "Решающее дерево")

plt.figure(figsize=(12, 8))
# Получим имена признаков после OneHotEncoder
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)
plot_tree(tree_clf, filled=True, feature_names=feature_names, class_names=['Не купили', 'Купили'])
plt.title("Визуализация дерева решений")
plt.show()

# Важность признаков
feature_importances = tree_clf.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Важность признаков')
plt.title('Важность признаков для дерева решений')
plt.show()

# Метод опорных векторов (SVM)
svm_clf = svm.SVC(kernel='linear', C=1, probability=True)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

print("\nМетод опорных векторов (SVM):")
evaluate_model(y_test, y_pred_svm)
plot_confusion_matrix(y_test, y_pred_svm, "Метод опорных векторов (SVM)")
plot_roc_curve(svm_clf, X_test, y_test, "Метод опорных векторов (SVM)")