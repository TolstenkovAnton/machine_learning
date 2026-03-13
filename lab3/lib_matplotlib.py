import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
df.replace(-200, np.nan, inplace=True)

# 1. Распределение CO(GT) по интервалам
co_bins = pd.cut(df['CO(GT)'].dropna(), bins=10)
co_count = co_bins.value_counts().sort_index()
co_count.plot(kind='bar')
plt.title('Распределение CO(GT) по интервалам')
plt.xlabel('Интервалы CO(GT)')
plt.ylabel('Количество наблюдений')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. То же с логарифмической Y
co_count.plot(kind='bar', logy=True)
plt.title('Распределение CO(GT) - логарифмическая Y')
plt.xlabel('Интервалы CO(GT)')
plt.ylabel('log(Количество)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3-5. Гистограммы T для CO выше/ниже среднего (density=True)
co_mean = df['CO(GT)'].mean()
high_co = df[df['CO(GT)'] > co_mean]['T'].dropna()
low_co = df[df['CO(GT)'] <= co_mean]['T'].dropna()

plt.figure(figsize=(10, 6))
plt.hist(high_co, bins=20, density=True, alpha=0.5, color='red', label='CO выше среднего')
plt.hist(low_co, bins=20, density=True, alpha=0.5, color='blue', label='CO ниже среднего')
plt.xlabel('Температура (°C)')
plt.ylabel('Плотность')
plt.title('Распределение температуры по уровню CO')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Время суток для C6H6
df['Time_clean'] = df['Time'].astype(str).str.replace('.', ':', regex=False)
df['Time'] = pd.to_datetime(df['Time_clean'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Time'].dt.hour


def time_period(hour):
    if pd.isna(hour):
        return 'Неизвестно'
    if 6 <= hour < 12:
        return 'Утро'
    elif 12 <= hour < 18:
        return 'День'
    elif 18 <= hour < 24:
        return 'Ночь'
    else:
        return 'Ночь'


df['Period'] = df['Hour'].apply(time_period)

c6h6_period = df.dropna(subset=['C6H6(GT)']).groupby('Period')['C6H6(GT)'].apply(list)

plt.figure(figsize=(10, 6))
for period in c6h6_period.index:
    if len(c6h6_period[period]) > 0:
        plt.hist(c6h6_period[period], alpha=0.7, label=period, bins=15)
plt.xlabel('C6H6(GT)')
plt.ylabel('Частота')
plt.title('Распределение C6H6 по времени суток')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Boxplot CO по времени суток
plt.figure(figsize=(12, 6))
df.boxplot(column='CO(GT)', by='Period', ax=plt.gca())
plt.title('CO(GT) по времени суток')
plt.suptitle('')  # Убираем подзаголовок
plt.xlabel('Время суток')
plt.ylabel('CO(GT)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Зависимость загрязнителей от температуры
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
pollutants = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
colors = ['red', 'blue', 'green', 'orange']

for i, (poll, color) in enumerate(zip(pollutants, colors)):
    ax = axes[i // 2, i % 2]

    valid_data = df[['T', poll]].dropna()
    ax.scatter(valid_data['T'], valid_data[poll], alpha=0.5, c=color, s=10)

    ax.set_xlabel('Температура (°C)')
    ax.set_ylabel(poll)
    ax.set_title(f'{poll} от T')

plt.tight_layout()
plt.show()
