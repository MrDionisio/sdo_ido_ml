# Домашняя работа 2
# выполнил работу: Кучиев Денис Юрьевич

import numpy as np

# Задание 1. Решение проблемы транслирования массивов разной формы

base_matrix = np.ones((3,2))  # Матрица 3x2
vector = np.arange(3)         # Вектор длины 3

# Решение через reshape
reshaped_vector = vector.reshape(3, 1)  # Преобразуем в матрицу 3x1
result1 = base_matrix + reshaped_vector
print("Результат с reshape:")
print(result1)

# Альтернативное решение через newaxis
newaxis_vector = vector[:, np.newaxis]
print("\nФорма вектора после newaxis:", newaxis_vector.shape)
result2 = base_matrix + newaxis_vector
print("Результат с newaxis:")
print(result2)

# Задание 2. Фильтрация элементов массива по условию

data_array = np.array([[1,2,3,4,5],
                       [6,7,8,9,10]])

# Общее количество элементов, удовлетворяющих условию
total_count = np.sum((data_array > 3) & (data_array < 9))
print("\nОбщее количество элементов 3 < x < 9:", total_count)

# Количество по строкам
row_counts = np.sum((data_array > 3) & (data_array < 9), axis=1)
print("Количество по строкам:", row_counts)

# Количество по столбцам
col_counts = np.sum((data_array > 3) & (data_array < 9), axis=0)
print("Количество по столбцам:", col_counts)