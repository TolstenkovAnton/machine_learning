import pandas as pd
import numpy as np


df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

# 2
print("\n2. head() и tail():")
print(df.head())
print(df.tail())

# 3
print("\n3. Шейп:", df.shape)

# 4
print("\n4. Признаки:", df.columns.tolist())

# 5
df.replace(-200, np.nan, inplace=True)
print("\n5. Пропуски:")
print(df.isnull().sum())

# 6
print("\n6. describe():")
print(df.describe())

# 7
print("\n7. info():")
df.info()

# 8
print("\n8. T уникальные:", df['T'].nunique())
print(df['T'].value_counts().head(10))

# 9
print("\n9. CO>3 и T<20:")
high_co_low_t = df[(df['CO(GT)'] > 3) & (df['T'] < 20)]
print(high_co_low_t.head())

# 10
df['NOx_CO_ratio'] = df['NOx(GT)'] / df['CO(GT)']

# 11
print("\n11. Новый шейп:", df.shape)

# 12
print("\n12. Чаще всего T:", df['T'].mode()[0])

# 13
print("\n13. Пропуски C6H6(GT):", df['C6H6(GT)'].isnull().sum())
print("Строки с пропусками:")
print(df[df['C6H6(GT)'].isnull()].head())

# 14
high_t = df[df['T'] > 25]
min_co_high_t = high_t['CO(GT)'].min()
print("\n14. Мин CO при T>25:", min_co_high_t)
print(high_t[high_t['CO(GT)'] == min_co_high_t].head(1))

# 15
high_rh = df[df['RH'] > 90]
print("\n15. RH>90:", len(high_rh))

# 16
co_high20 = df[df['T'] > 20]['CO(GT)'].mean()
co_low20 = df[df['T'] <= 20]['CO(GT)'].mean()
diff_co = round(co_high20 - co_low20, 2)
print("\n16. Разница CO (выше20 - ниже20):", diff_co)

# 17
ozone_mean = df['PT08.S5(O3)'].mean()
df['High_Ozone'] = (df['PT08.S5(O3)'] > ozone_mean).astype(int)
print("\n17. High_Ozone создан.\n1:", df['High_Ozone'].sum())

# 18
print("\n18. NOx модa:", df['NOx(GT)'].mode()[0])

# 19
c6_mean = df['C6H6(GT)'].mean()
nox_mean = df['NOx(GT)'].mean()
both_high = ((df['C6H6(GT)'] > c6_mean) & (df['NOx(GT)'] > nox_mean)).sum()
print("\n19. Оба > ср.:", both_high)

# 20
low_no2 = df[df['NO2(GT)'] < 50]
max_t_low_no2 = low_no2['T'].max()
print("\n20. Макс T при NO2<50:", max_t_low_no2)

# 21
co_mean = df['CO(GT)'].mean()
high_co_all = df[df['CO(GT)'] > co_mean]
print("\n21. CO > ср.:", len(high_co_all))
print(high_co_all.head())

# 22
t_mean = df['T'].mean()
high_t_c6 = df[df['T'] > t_mean]['C6H6(GT)'].mean()
low_t_c6 = df[df['T'] <= t_mean]['C6H6(GT)'].mean()
print("\n22. C6H6 выше ср.T:", round(high_t_c6, 2))
print("     C6H6 ниже ср.T:", round(low_t_c6, 2))
