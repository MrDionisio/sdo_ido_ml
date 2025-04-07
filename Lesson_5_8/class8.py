#Кучиев Денис Юрьевич 
# Конспект №8
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl

fig = plt.figure()

ax = plt.axes(projection="3d")

z1 = np.linspace(0,15, 100)
y1 = np.cos(z1)
x1 = np.sin(z1)


ax.plot3D(x1,y1,z1,'green')

z2 = 15*np.random.random(100)
y2 = np.cos(z2)+0.1*np.random.random(100)
x2 = np.sin(z2) + 0.1 * np.random.random(100)

ax.scatter3D(x2, y2, z2, c=z2, cmap='Greens')
plt.show()

def f(x,y):
    return np.sin(np.sqrt(x**2+y**2))

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

ax.contour3D(X, Y, Z, 40, cmap='binary')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(45, 45)

plt.show()
ax=plt.axes(projection='3d')

ax.scatter3D(X, Y, Z, c=Z, cmap='Greens')
# Каркасный
ax.plot_wireframe(X, Y, Z)

# Поверхностный график
ax.plot_surface(X, Y, Z, cmap='viridis')

plt.show()

ax = plt.axes(projection='3d')

r = np.linspace(0,6,20)
theta = np.linspace(-0.9*np.pi, 0.8*np.pi, 40)

r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)

Y = r* np.cos(theta)

Z = f(X, Y)

ax.plot_surface(X, Y, Z,rstride = 1, cstride=1,  cmap='viridis')

plt.show()

# Триангуляция поверхности

ax = plt.axes(projection='3d')

theta = 2 * np.pi + np.random.random(10000)
r = 6 * np.random.random(10000)

x = r*np.sin(theta)
y = r*np.cos(theta)

z = f(x, y)
#ax.scatter(x, y, z, c=z, cmap='viridis')

ax.plot_trisurf(x, y, z, cmap='viridis')

ax.view_init(45, 60)
plt.show()
# Seaborn

data = np.random.multivariate_normal([0,0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
fig = plt.figure()
plt.hist(data['x'], alpha=0.5)
plt.hist(data['y'], alpha=0.5)

plt.show()

fig = plt.figure()
sns.kdeplot(data=data, fill=True)

iris = sns.load_dataset('iris')
print(iris)

sns.pairplot(iris,hue='species', height=2.5)

plt.show()


# 
tips = sns.load_dataset('tips')

grid = sns.FacetGrid(tips, row='sex', col='day', hue='time')
grid.map(plt.hist, 'tip', bins=10)

plt.show()

sns.catplot(data=tips, x='day', y='total_bill', kind='box' )


plt.show()
sns.jointplot(data=tips, x='tip', y='total_bill', kind='hex')

plt.show()
planets=sns.load_dataset('planets')
sns.catplot(data=planets, x='year', kind='count', order=range(2005, 2015), hue='method')
# Числовые пары
sns.pairplot(tips)

plt.show()
# Тепловая карта
tips_corr = tips[['total_bill', 'tip', 'size']]
sns.heatmap(tips_corr.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1)

#0 - независимы
# 1 - чем больше одно тем больше другое
# -1 обратная пропорциональность

plt.show()
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
sns.regplot(data=tips, x='total_bill', y='tip')
sns.relplot(data=tips, x='total_bill', y='tip', hue='sex')
sns.lineplot(data=tips, x='total_bill', y='tip', hue='sex')
# Сравнение числовых и категориальных данных
# Гистограмма

sns.barplot(data=tips, x='day', y='total_bill',hue='sex')
sns.pointplot(data=tips, x='day', y='total_bill',hue='sex')
# Ящик с усами
sns.boxplot(data=tips, x='day', y='total_bill',hue='sex')
# Скрипичная 
sns.violinplot(data=tips, x='day', y='total_bill',hue='sex')
# Одномерная д. рассеяния
sns.stripplot(data=tips, x='day', y='total_bill',hue='sex')