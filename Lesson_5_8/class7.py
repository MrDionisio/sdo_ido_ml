#Кучиев Денис Юрьевич
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
fig, ax = plt.subplots(2,3)


for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i,j)), fontsize=16, ha='center')


plt.show()
fig, ax = plt.subplots(2,3, sharex='col', sharey='row')


for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i,j)), fontsize=16, ha='center')


plt.show()
grid = plt.GridSpec(2,3)

plt.subplot(grid[0:,0])
plt.subplot(grid[0,1:])
plt.subplot(grid[1,2])
plt.subplot(grid[1,1:2])


plt.show()
rng = np.random.default_rng(1)
mean = [0,0]
cov = [[1,1], [1,2]]

x,y = rng.multivariate_normal(mean=mean, cov=cov, size=3000).T

fig = plt.figure()
grid = plt.GridSpec(4,4,hspace=0.4, wspace=0.4)

main_ax = fig.add_subplot(grid[:-1, 1:])

main_ax.plot(x,y, 'ok', markersize=3, alpha=0.2)

y_his = fig.add_subplot(grid[:-1, 0], sharey=main_ax, xticklabels=[])

x_his = fig.add_subplot(grid[-1, 1:], sharex=main_ax, yticklabels=[])

y_his.hist(y, 40, orientation='horizontal',color='grey' , histtype='stepfilled')
x_his.hist(x, 40,color='grey' , histtype='step')
plt.show()

## Поясняющие надписи

births = pd.read_csv("./births-1969.csv")

births.index=pd.to_datetime(10000*births.year+100*births.month+births.day, format='%Y%m%d')

#print(births.head())


births_by_date=births.pivot_table('births', [births.index.month, births.index.day])


births_by_date.index=[
    datetime(1969,month,day) for (month, day) in births_by_date.index
]

fig, ax = plt.subplots()

births_by_date.plot(ax=ax)

style = dict(size=10, color='gray')
ax.text('1969-01-01', 5500, "Новый год", **style)
ax.text('1969-09-01', 4500, "День знаний")

ax.set(title='Рождаемость в 1969 году', ylabel='Число рождений')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))  
ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=15))

plt.show()
fig = plt.figure()
ax1=plt.axes()
ax2=plt.axes([0.4, 0.3, 0.1, 0.2])

ax1.set_xlim(0,2)

ax1.text(0.6, 0.8, "Data1 (0.6 0.8)", transform=ax2.transData)
ax1.text(0.6, 0.8, "Data1 (0.6 0.8)", transform=ax1.transData)
plt.show()


# подписи на тексте
fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('Локальный максимум', xy = (6.28, 1), xytext=(10,4), arrowprops=dict(facecolor='red'))
ax.annotate('Локальный минимум', xy = (3.14, -1), xytext=(2,-6), arrowprops=dict(facecolor='blue', arrowstyle='->'))

plt.show()



# выбор, сколько делений на графике делать
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)

for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(5))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))

plt.show()



x = np.random.randn(1000)

fig = plt.figure(facecolor='gray')
ax = plt.axes(facecolor='green')

plt.grid(color='w', linestyle='solid')

ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

plt.style.use('default')
plt.hist(x)

plt.show()

# .matplotlibrc - файл, в нем можно сохранить, как будет выглядеть ваш график, и потом использовать