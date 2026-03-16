import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# 1) ЗАГРУЗКА ДАННЫХ
print("1. Загрузка данных...")
df = pd.read_csv('Customers.csv', sep=',')
print(f"Форма датасета: {df.shape}")
print(f"Столбцы: {list(df.columns)}")
print(df.head(3), "\n")

# 2) ПЕРВЫЕ И ПОСЛЕДНИЕ 10 СТРОК
print("2. Первые и последние 10 строк:")
print("head(10):")
print(df.head(10))
print("\ntail(10):")
print(df.tail(10))
df.head(10).to_csv('head10.csv', index=False)  # Для отчета[code_file:22]
df.tail(10).to_csv('tail10.csv', index=False)  # Для отчета[code_file:23]

# 3) СТАТИСТИКА ПО ЗНАЧЕНИЯМ
print("\n3. Статистика по признакам:")
print(df.describe())
df.describe().to_csv('describe.csv')  # Для отчета[code_file:21]

# 4) ПОДРОБНОЕ ОПИСАНИЕ
print("\n4. Подробное описание (info):")
df.info()

# 5) ГЕНДЕРНОЕ РАСПРЕДЕЛЕНИЕ
print("\n5. Гендерное распределение:")
gender_counts = df['Gender'].value_counts()
print(gender_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Столбиковая
gender_counts.plot(kind='bar', ax=ax1, color=['lightblue', 'pink'])
ax1.set_title('Гендерное распределение (столбцы)')
ax1.set_ylabel('Количество')
ax1.tick_params(axis='x', rotation=0)

# Круговая
ax2.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
ax2.set_title('Гендерное распределение (круг)')

plt.tight_layout()
plt.savefig('gender_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 6) РАСПРЕДЕЛЕНИЕ ВОЗРАСТОВ
print("\n6. Распределение возрастов:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Гистограмма
df['Age'].hist(bins=20, ax=ax1, edgecolor='black', alpha=0.7, color='skyblue')
ax1.set_title('Гистограмма возрастов')
ax1.set_xlabel('Возраст (лет)')
ax1.set_ylabel('Частота')

# Boxplot
df.boxplot(column='Age', ax=ax2, patch_artist=True)
ax2.set_title('Ящик с усами (возраст)')
ax2.set_ylabel('Возраст (лет)')

plt.tight_layout()
plt.savefig('age_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 7) ГОДОВОЙ ДОХОД
print("\n7. Годовой доход:")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Гистограмма
df['Annual Income ($)'].hist(bins=20, ax=ax1, edgecolor='black', alpha=0.7, color='lightgreen')
ax1.set_title('Гистограмма годового дохода')
ax1.set_xlabel('Доход (тыс. $)')

# Плотность
df['Annual Income ($)'].plot(kind='density', ax=ax2, color='green', lw=2)
ax2.set_title('График плотности дохода')
ax2.set_xlabel('Доход (тыс. $)')

plt.tight_layout()
plt.savefig('income_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 8) АНАЛИЗ РАСХОДОВ (Spending Score)
print("\n8. Анализ расходов клиентов:")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Гистограмма
df['Spending Score (1-100)'].hist(bins=20, ax=axes[0,0], edgecolor='black', alpha=0.7, color='salmon')
axes[0,0].set_title('Гистограмма трат')
axes[0,0].set_xlabel('Spending Score')

# Boxplot
df.boxplot(column='Spending Score (1-100)', ax=axes[0,1], patch_artist=True)
axes[0,1].set_title('Ящик с усами (траты)')

# Плотность
df['Spending Score (1-100)'].plot(kind='density', ax=axes[1,0], color='red', lw=2)
axes[1,0].set_title('Плотность трат')
axes[1,0].set_xlabel('Spending Score')

# Violin plot
sns.violinplot(data=df, y='Spending Score (1-100)', ax=axes[1,1], color='lightcoral')
axes[1,1].set_title('Violin plot трат')

plt.tight_layout()
plt.savefig('spending_charts.png', dpi=300, bbox_inches='tight')
plt.show()

# 9) OПТИМАЛЬНОЕ КОЛИЧЕСТВО КЛАСТЕРОВ (KMeans)
print("\n9. Определение оптимального k (KMeans):")
X = df[['Annual Income ($)', 'Spending Score (1-100)']]  # Стандартные признаки для сегментации[web:18]

inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_init=10, random_state=42, n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_title('Метод локтя (Elbow Method)')
ax1.set_xlabel('Количество кластеров (k)')
ax1.set_ylabel('Inertia')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, sil_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_title('Silhouette Score')
ax2.set_xlabel('Количество кластеров (k)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

optimal_k = K_range[np.argmax(sil_scores)]
print(f"Оптимальное k по Silhouette: {optimal_k} (max score: {max(sil_scores):.3f})")

# 10) ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ
print("\n10. Кластеризация (k=5):")
kmeans = KMeans(n_init=10, random_state=42, n_clusters=5)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(data=df, x='Annual Income ($)', y='Spending Score (1-100)',
                          hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Результаты кластеризации KMeans (k=5)\n(Annual Income vs Spending Score)', fontsize=14)
plt.xlabel('Годовой доход (тыс. $)')
plt.ylabel('Оценка трат (1-100)')
plt.legend(title='Кластер')
plt.grid(True, alpha=0.3)

# Центры кластеров
centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
sns.scatterplot(x=centers['Annual Income ($)'], y=centers['Spending Score (1-100)'],
                color='red', s=300, marker='X', ax=plt.gca(), label='Центры')
plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nЦентры кластеров:")
print(centers.round(2))

# Сводка по кластерам
print("\nРаспределение по кластерам:")
print(df['Cluster'].value_counts().sort_index())
