# Конспект лекции 3
# сделал работу: Кучиев Денис Юрьевич

import numpy as np
import pandas as pd

# Pandas расширяет возможности numpy, позволяя работать с таблицами, где строки и столбцы можно индексировать не только числами, но и осмысленными метками.

# Основные структуры: Series, DataFrame, Index

## Series - одномерный массив с индексами

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(type(data))

# Доступ к значениям и индексам
print(data.values)  # сами значения
print(type(data.values))  # numpy-массив
print(data.index)  # индексы по умолчанию
print(type(data.index))  # pandas RangeIndex

# Можно обращаться по индексу и делать срезы
print(data[0])
print(data[1:3])  # обрезаем по диапазону

# Индексы можно задавать явно

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)
print(data['a'])
print(data['b':'d'])  # В отличие от обычных срезов, здесь включается последний элемент

print(type(data.index))  # теперь это просто Index, а не RangeIndex

# Индексы могут быть любого типа

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[1, 10, 7, 'd'])
print(data)
print(data[1])
print(data[10:'d'])

# Другой способ создания Series - через словарь

population_dict = {
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_4': 1004,
    'city_5': 1005,
}

population = pd.Series(population_dict)
print(population)
print(population['city_4'])
print(population['city_4':'city_5'])

# Способы создания Series:
# - список Python
# - numpy-массив
# - скалярное значение
# - словарь

## DataFrame - двумерный массив с индексами

# DataFrame можно собрать из нескольких Series

area_dict = {
    'city_1': 901,
    'city_2': 11,
    'city_3': 103,
    'city_4': 105,
    'city_5': 10011,
}

area = pd.Series(area_dict)

states = pd.DataFrame({
    'population': population,
    'area': area
})

print(states)

# Доступ к внутренним элементам
print(states.values)  # массив значений
print(states.index)  # индексы строк
print(states.columns)  # индексы столбцов

# Доступ к столбцам
print(states['area'])

# DataFrame можно создать разными способами:
# - из нескольких Series
# - из списка словарей
# - из словаря Series
# - из numpy-массива
# - из структурированного numpy-массива

## Index - механизм работы с индексами в Series и DataFrame

ind = pd.Index([2, 3, 5, 7, 11])
print(ind[1])  # доступ по индексу
print(ind[::2])  # срезы

# Index ведёт себя как множество
indA = pd.Index([1, 2, 3, 4, 5])
indB = pd.Index([2, 3, 4, 5, 6])
print(indA.intersection(indB))  # пересечение

## Выборка данных из Series

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data.keys())  # список индексов
print(list(data.items()))  # превращение в список пар (индекс, значение)

# Изменение значений
data['a'] = 100
data['z'] = 1000  # добавление нового элемента
print(data)

# Доступ по индексаторам
print(data.loc['a'])  # доступ по метке
print(data.iloc[1])  # доступ по порядковому номеру

## DataFrame: доступ по индексаторам

print(states.iloc[:3, 1:2])  # доступ по номерам
print(states.loc[:'city_4', 'population'])  # доступ по меткам
print(states.loc[states['population'] > 1002, ['area', 'population']])

## Универсальные функции

rng = np.random.default_rng()
s = pd.Series(rng.integers(0, 10, 4))
print(np.exp(s))  # применение функции к каждому элементу

## Объединение данных

pop = pd.Series({'city_1': 1001, 'city_2': 1002, 'city_3': 1003, 'city_41': 1004, 'city_51': 1005})
area = pd.Series({'city_1': 901, 'city_2': 11, 'city_3': 103, 'city_42': 105, 'city_52': 10011})

# Если объединять Series с разными индексами, появятся NaN

data = pd.DataFrame({'area': area, 'population': pop})
print(data)

# Заполняем NaN единицами
print(data.fillna(1))

## Транспонирование

df = pd.DataFrame(rng.integers(0, 10, (3, 4)), columns=['a', 'b', 'c', 'd'])
print(df.T)  # транспонирование

# Вычитание с учетом индексов
print(df - df.iloc[0])

## Работа с пропущенными значениями

x = pd.Series([1, 2, 3, np.nan, None, pd.NA])
print(x.isnull())  # проверяем, где NaN
print(x.dropna())  # удаляем пропущенные значения

# Для DataFrame аналогично

df = pd.DataFrame([
    [1, 2, 3, np.nan, None, pd.NA],
    [1, 2, 3, None, 5, 6],
    [1, np.nan, 3, None, np.nan, 6],
])
print(df.dropna(axis=1, how='all'))  # удаляем столбцы, где все NaN
print(df.dropna(axis=1, thresh=2))  # оставляем, если хотя бы 2 ненулевых

# Заполнение значений
print(df.fillna(method='ffill'))  # заполняем сверху вниз
print(df.fillna(method='bfill'))  # заполняем снизу вверх
