
# выполнил работу: Кучиев Денис Юрьевич
import numpy as np
import matplotlib.pyplot as plt

# Генерация случайных данных
rng = np.random.default_rng(1)
random_data = rng.random(50)

print(random_data)
print(sum(random_data))
print(np.sum(random_data))

sample_array = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10]
])

print(np.sum(sample_array))
print(np.sum(sample_array, axis=0))
print(np.sum(sample_array, axis=1))

print(np.min(sample_array))
print(np.min(sample_array, axis=0)) 
print(np.min(sample_array, axis=1))

print(sample_array.min())
print(sample_array.min(0))
print(sample_array.min(1))

print(np.nanmin(sample_array))
print(np.nanmin(sample_array, axis=0)) 
print(np.nanmin(sample_array, axis=1))

# Операции с массивами
arr1 = np.array([0,1,2])
arr2 = np.array([5,5,5])

print(arr1 + arr2)
print(arr1 + 5)

matrix = np.array([[0,1,2], [3,4,5]])
print(matrix + 5)

vec1 = np.array([0,1,2])
vec2 = np.array([[0],[1],[2]])
print(vec1 + vec2)

# Правила broadcasting
matrix_a = np.array([[0,1,2], [3,4,5]])
scalar_b = np.array([5])

print(matrix_a.ndim, matrix_a.shape)
print(scalar_b.ndim, scalar_b.shape)
print(matrix_a + scalar_b)

ones_matrix = np.ones((2,3))
range_vec = np.arange(3)

result = ones_matrix + range_vec
print(result)

# Центрирование данных
data_matrix = np.array([
    [1,2,3,4,5,6,7,8,9],
    [9,8,7,6,5,4,3,2,1]
])

col_means = data_matrix.mean(0)
centered_cols = data_matrix - col_means
print(centered_cols)

row_means = data_matrix.mean(1)
row_means = row_means[:, np.newaxis]
centered_rows = data_matrix - row_means
print(centered_rows)

# Визуализация
x_vals = np.linspace(0, 5, 50)
y_vals = np.linspace(0, 5, 50)[:, np.newaxis]

z_vals = np.sin(x_vals)**3 + np.cos(20+y_vals*x_vals) * np.sin(y_vals)

plt.imshow(z_vals)
plt.colorbar()
plt.show()

# Логические операции
nums = np.array([1,2,3,4,5])
matrix_nums = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(nums < 3)
print(np.sum(nums < 4))
print(np.sum(matrix_nums < 4, axis=0))
print(np.sum(matrix_nums < 4, axis=1))

# Маскирование
print(nums[nums < 3])

# Бинарные операции
print(bin(42 & 59))

# Индексация
sequence = np.array([0,1,2,3,4,5,6,7,8,9])
indices = [1,5,7]
print(sequence[indices])

# Модификация через индексы
sequence_copy = np.arange(10)
mod_indices = np.array([2,1,8,4])
sequence_copy[mod_indices] = 999
print(sequence_copy)

# Сортировка
unsorted = [3,2,4,5,6,1,4,1,7,8]
print(np.sort(unsorted))

# Структурированные данные
structured_data = np.zeros(4, dtype={
    'names': ('name', 'age'),
    'formats': ('U10', 'i4')
})

names = ['name1', 'name2', 'name3', 'name4']
ages = [10, 20, 30, 40]

structured_data['name'] = names
structured_data['age'] = ages

print(structured_data[structured_data['age'] > 20]['name'])

# Массивы записей
record_array = structured_data.view(np.recarray)
print(record_array[-1].name)
