"""
Лабораторная работа №5: Классификация грибов (mushrooms.csv) - ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ
Исправлена ошибка ConfusionMatrixDisplay.plot()
"""

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
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]

df = pd.read_csv('mushrooms.csv', header=None, names=columns, sep=',')
df = df.replace('?', 'missing')

print(f"Форма: {df.shape}")
print("Class распределение:\n", df['class'].value_counts())


def prepare_data(df, target_col, additional_drop=None):
    """Очистка + Label Encoding"""
    drop_cols = [target_col]
    if additional_drop:
        drop_cols.extend(additional_drop if isinstance(additional_drop, list) else [additional_drop])

    X = df.drop(drop_cols, axis=1).copy()
    y = df[target_col].copy()

    # Удаление редких классов (<10)
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 10].index
    if len(rare_classes) > 0:
        print(f"Удаляем редкие классы: {list(rare_classes)}")
        mask = ~y.isin(rare_classes)
        X = X[mask]
        y = y[mask]

    # Label Encoding
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    print(f"После очистки: {X.shape[0]} строк, {len(np.unique(y))} классов")
    return X, y


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except:
        print("  Используем split без stratify")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, is_binary=False):
    """Обучение + метрики"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    avg = 'binary' if is_binary else 'macro'
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average=avg, zero_division=0),
        'Recall': recall_score(y_test, y_pred, average=avg, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, average=avg, zero_division=0)
    }

    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # AUC-ROC
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
                print(f"  AUC-ROC: {auc:.4f}")
            else:
                print("  AUC-ROC: многоклассовая (пропущено)")
        except Exception as e:
            print(f"  AUC-ROC: ошибка {e}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Размер CM: {cm.shape}")

    # Способ 1: seaborn heatmap (НАДЕЖНЫЙ)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(cm.shape[1]),
                yticklabels=range(cm.shape[0]))
    plt.title(f'Матрица ошибок — {model_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    # Дерево решений
    if isinstance(model, DecisionTreeClassifier):
        plt.figure(figsize=(20, 8))
        plot_tree(model, feature_names=X_train.columns.tolist(),
                  class_names=[f'C{i}' for i in range(len(np.unique(y_train)))],
                  filled=True, rounded=True, max_depth=3)
        plt.title(f'Дерево решений — {model_name}')
        plt.show()


# МОДЕЛИ
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=50),
    'NaiveBayes': CategoricalNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("\n=== 1. СЪЕДОБНОСТЬ ГРИБОВ (class) ===")
X, y = prepare_data(df, 'class')
X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
for name, model in models.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=True)

print("\n=== 2. МЕСТООБИТАНИЕ (habitat) ===")
X, y = prepare_data(df, 'habitat')
X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
for name, model in models.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, name, is_binary=False)

print("\n=== 3. GILL ПРИЗНАКИ ===")

# Gill-attachment
print("\n3.1 gill-attachment")
X, y = prepare_data(df, 'gill-attachment')
X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
for name, model in models.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, f"{name} (gill-att)", is_binary=False)

# Gill-spacing
print("\n3.2 gill-spacing")
X, y = prepare_data(df, 'gill-spacing')
X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
for name, model in models.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, f"{name} (gill-sp)", is_binary=False)

# Gill-size (бинарная)
print("\n3.3 gill-size")
X, y = prepare_data(df, 'gill-size')
X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
for name, model in models.items():
    train_and_evaluate(model, X_train, X_test, y_train, y_test, f"{name} (gill-size)", is_binary=True)
