# домашняя работа 3
# выполнил: кучиев денис юрьевич

import numpy as np
import pandas as pd

# задание 1. способы создания series

# список python
d1 = [1, 2, 3, 4]
ser1 = pd.Series(d1)
print('series список python')
print(ser1)

# массив numpy
d2 = np.array([1, 2, 3, 4])
ser2 = pd.Series(d2)
print('series массив numpy')
print(ser2)

# скалярные значения
d3 = 123
ser3 = pd.Series(d3, index=[0, 1, 2, 3])
print('series скалярные значения')
print(ser3)

# словари
d4 = {"d1": 1, "d2": 2, "d3": 3, "d4": 4}
ser4 = pd.Series(d4)
print('series словари')
print(ser4)

# задание 2. способы создания dataframe

# через объекты series
ser1 = pd.Series([1, 2, 3, 4])
ser2 = pd.Series([5, 6, 7, 8])
data1 = pd.DataFrame([ser1, ser2])
print('dataframe через объекты series')
print(data1)

# списки словарей
l1 = [{'a1': 1, 'b1': 2}, {'a1': -1, 'b1': -2}]
data2 = pd.DataFrame(l1)
print('dataframe через списки словарей')
print(data2)

# словари объектов series
d1 = {'col1': ser1, 'col2': ser2}
data3 = pd.DataFrame(d1)
print('dataframe через словари объектов series')
print(data3)

# двумерный массив numpy
a1 = np.array([[1, 2, 3], [4, 5, 6]])
data4 = pd.DataFrame(a1)
print('dataframe через двумерный массив numpy')
print(data4)

# структурированный массив numpy
a2 = np.array([('city_1', 1009), ('city_2', 2009)], dtype=[('city', 'U10'), ('pop', 'i4')])
data5 = pd.DataFrame(a2)
print('dataframe через структурированный массив numpy')
print(data5)

# задание 3. объединение двух series с разными ключами
pop = pd.Series({'city_1': 1001, 'city_2': 1002, 'city_3': 1003, 'city_41': 1004, 'city_51': 1005})
area = pd.Series({'city_1': 901, 'city_2': 11, 'city_3': 103, 'city_42': 105, 'city_52': 10011})
data = pd.DataFrame({'area1': area, 'pop1': pop}).fillna(1)
print(data)

# задание 4. вычитание по столбцам в dataframe
rng = np.random.default_rng(1)
A = rng.integers(0, 10, (3, 4))
df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print('исходный df')
print(df)

print('df вычитание по столбцам (способ 1)')
print(df - df.iloc[:, 0].values[:, np.newaxis])

print('df вычитание по столбцам (способ 2)')
print(df.sub(df.iloc[:, 0], axis=0))

# задание 5. использование ffill() и bfill()
df = pd.DataFrame({'d1': [1, np.nan, 3, np.nan, 10],
                    'd2': [np.nan, 4, np.nan, 5, 11],
                    'd3': [6, np.nan, 7, np.nan, 12]})
print('исходный df')
print(df)

print('df при использовании ffill()')
print(df.ffill())

print('df при использовании ffill() по строкам')
print(df.ffill(axis=1))

print('df при использовании bfill()')
print(df.bfill())

print('df при использовании bfill() по строкам')
print(df.bfill(axis=1))