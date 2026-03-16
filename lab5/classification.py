import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import warnings

warnings.filterwarnings('ignore')

columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]

df = pd.read_csv('mushrooms.csv',
                 header=None,
                 names=columns,
                 sep=',')

df = df.replace('?', 'missing')

def prepare_data(df, target_col, additional_drop=None):
    """Подготовка данных: Label Encoding для всех категориальных признаков"""
    drop_cols = [target_col]
    if additional_drop:
        drop_cols.extend(additional_drop if isinstance(additional_drop, list) else [additional_drop])

    X = df.drop(drop_cols, axis=1)
    y = df[target_col].copy()

    # Label Encoding признаков
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Label Encoding целевой переменной
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    return X, y, le_y.classes_


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, is_binary=False):
    """Обучение модели + расчёт всех метрик + визуализация"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Метрики
    avg = 'binary' if is_binary else 'macro'
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

    print(f"\n{model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    # AUC-ROC
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if is_binary:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
        print(f"AUC-ROC: {auc:.4f}")
    else:
        print("AUC-ROC: не поддерживается моделью")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Матрица ошибок — {model_name}')
    plt.show()

    # Визуализация дерева решений (только для DecisionTree)
    if isinstance(model, DecisionTreeClassifier):
        plt.figure(figsize=(20, 10))
        plot_tree(model,
                  feature_names=X_train.columns,
                  class_names=[str(cls) for cls in np.unique(y_train)],
                  filled=True,
                  rounded=True,
                  max_depth=5)  # ограничиваем для читаемости
        plt.title(f'Дерево решений — {model_name}')
        plt.show()

# 1. Предсказание съедобности (Class: e/p) — бинарная классификация
print("=== 1. Предсказание съедобности гриба (Class) ===")

X, y, _ = prepare_data(df, 'class')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models_task1 = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

for name, model in models_task1.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=True)

# 2. Предсказание типа местообитания (Habitat) — многоклассовая классификация
print("\n=== 2. Предсказание типа местообитания (Habitat) ===")

X, y, _ = prepare_data(df, 'habitat')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models_task2 = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naive Bayes (Categorical)': CategoricalNB()
}

for name, model in models_task2.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=False)

# 3. Предсказание структуры пластинок (Gill-*)
print("\n=== 3. Предсказание структуры пластинок ===")

# 3.1 Gill-attachment (многоклассовая)
print("\n3.1 Gill-attachment")
X, y, _ = prepare_data(df, 'gill-attachment')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models_task3 = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naive Bayes (Categorical)': CategoricalNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

for name, model in models_task3.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=False)

# 3.2 Gill-spacing (многоклассовая)
print("\n3.2 Gill-spacing")
X, y, _ = prepare_data(df, 'gill-spacing')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for name, model in models_task3.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=False)

# 3.3 Gill-size (бинарная)
print("\n3.3 Gill-size")
X, y, _ = prepare_data(df, 'gill-size')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for name, model in models_task3.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=True)

# 3.4 Комбинированная целевая переменная (все три gill-признака вместе)
print("\n3.4 Комбинированная классификация (Gill-attachment + spacing + size)")
df_combined = df.copy()
df_combined['gill_combined'] = (df_combined['gill-attachment'] + '-' +
                                df_combined['gill-spacing'] + '-' +
                                df_combined['gill-size'])

X, y, _ = prepare_data(
    df_combined,
    target_col='gill_combined',
    additional_drop=['gill-attachment', 'gill-spacing', 'gill-size']
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Количество уникальных комбинаций: {len(np.unique(y))}")

for name, model in models_task3.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name + " (combined)", is_binary=False)
