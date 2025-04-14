#Наивный байесовская классификация
# Набор моделей, которые предлагают быстрые и просты алгоритмы классификации
# Подходят для данных с большой размерность и с маленьким количество гиперпараметров, подходит для начального приближения.



#Гауссовский наивный байесовский классификатор

# Наивность состоит в том, что данные всех категорий взяты из простого нормального распределения и они не зависят друг от друга

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

iris = sns.load_dataset('iris')


sns.pairplot(iris, hue='species')

plt.show()

data = iris[['sepal_length', 'petal_length', 'species']]

#setosa - versicolor
print(data.shape)
data_df = data[data['species'] != "virginica"]
print(data_df.shape)

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

model = GaussianNB()
model.fit(X, Y)

print(model.theta_[0], model.var_[0])
print(model.theta_[1], model.var_[1])

data_df_setosa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])

plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])


x1_p = np.linspace(min(data_df['sepal_length']),max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']),max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p =pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, 
    columns=['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)


X_p['species'] = y_p

print(X_p.head())

X_p_setosa = X_p[X_p['species']=='setosa']
X_p_versicolor = X_p[X_p['species']!='setosa']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)


theta0 = model.theta_[0]
var0 = model.var_[0]

theta1 = model.theta_[1]
var1 = model.var_[1]

z1 = 1 / (2*np.pi * (var0[0]*var0[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta0[0]) ** 2 / (var0[0]) + (X2_p - theta0[1]) ** 2 / (var0[1])))

z2 = 1 / (2*np.pi * (var1[0]*var1[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta1[0]) ** 2 / (var1[0]) + (X2_p - theta1[1]) ** 2 / (var1[1])))

plt.contour(X1_p, X2_p, z1)

plt.contour(X1_p, X2_p, z2)

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X1_p, X2_p, z1)  
ax.contour3D(X1_p, X2_p, z2)  



plt.show()

#versicolor - virginica
data = iris[['sepal_length', 'petal_length', 'species']]

#setosa - versicolor
print(data.shape)
data_df = data[data['species'] != "setosa"]
print(data_df.shape)

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

model = GaussianNB()
model.fit(X, Y)

print(model.theta_[0], model.var_[0])
print(model.theta_[1], model.var_[1])

data_df_setosa = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])

plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])


x1_p = np.linspace(min(data_df['sepal_length']),max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']),max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p =pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, 
    columns=['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)


X_p['species'] = y_p

print(X_p.head())

X_p_setosa = X_p[X_p['species']=='virginica']
X_p_versicolor = X_p[X_p['species']!='setosa']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)


theta0 = model.theta_[0]
var0 = model.var_[0]

theta1 = model.theta_[1]
var1 = model.var_[1]

z1 = 1 / (2*np.pi * (var0[0]*var0[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta0[0]) ** 2 / (var0[0]) + (X2_p - theta0[1]) ** 2 / (var0[1])))

z2 = 1 / (2*np.pi * (var1[0]*var1[1]) ** 0.5) * np.exp(-0.5 * ((X1_p - theta1[0]) ** 2 / (var1[0]) + (X2_p - theta1[1]) ** 2 / (var1[1])))

plt.contour(X1_p, X2_p, z1)

plt.contour(X1_p, X2_p, z2)

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X1_p, X2_p, z1)  
ax.contour3D(X1_p, X2_p, z2)  



plt.show()

#versicolor - virginica