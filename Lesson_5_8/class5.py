import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)

fig = plt.figure()
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

#fig.savefig('1.png')
print(fig.canvas.get_supported_filetypes())

# matlab стиль
plt.Figure()

plt.subplot(2,1,1)
x=np.linspace(0,10,100)
plt.plot(x, np.sin(x))

plt.subplot(2,1,2)
plt.plot(x, np.cos(x))
plt.show()
#ООП стиль

fig, ax = plt.subplots(2)

ax[0].plot(x, np.sin(x), color='blue')
ax[1].plot(x, np.cos(x))

# Цвета линий
# - blue
# - rgbcmyk -> rg
# - 0.14 - градация серого
# RRGGBB - FF00EE
# RGB - (1.0, 0.2, 0.3)
# HTML - 'salmon'

ax[0].plot(x, np.sin(x), color='blue', linestyle= 'solid')
ax[0].plot(x, np.sin(x+4), color='g', linestyle= 'dashed')
ax[0].plot(x, np.sin(x+3), color='0.75', linestyle= '-.')
ax[0].plot(x, np.sin(x+2), color='#FF00EE', linestyle= ':')
ax[0].plot(x, np.sin(x+1), '--k')

# Стиль линии linestyle
# - сплошная '-', 'solid'
# '--', 'dashed'
# '-.', 'dashdot'
# ':' , 'dotted'



fig, ax = plt.subplots(4)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.sin(x))
ax[2].plot(x, np.sin(x))
ax[3].plot(x, np.sin(x))

ax[1].set_xlim(-2, 12)
ax[1].set_ylim(-1.5, 1.5)
ax[2].set_xlim(12, -2)
ax[2].set_ylim(1.5, -1.5)

ax[3].autoscale(tight=True)

plt.show()
plt.subplot(3,1,1)
plt.plot(x, np.sin(x))

plt.title("Sin")
plt.xlabel("x")
plt.ylabel("Sin(X)")

plt.subplot(3,1,2)
plt.plot(x, np.sin(x), '-g', label="sin(x)")
plt.plot(x, np.cos(x), '-b', label="cos(x)")
plt.legend()
plt.title("Sin")
plt.xlabel("x")
plt.ylabel("Sin(X)")

plt.subplot(3,1,3)
plt.plot(x, np.sin(x), '-g', label="sin(x)")
plt.plot(x, np.cos(x), '-b', label="cos(x)")
plt.legend()
plt.title("Sin")
plt.xlabel("x")
plt.ylabel("Sin(X)")

plt.axis('equal')

plt.subplots_adjust(hspace = 1)

x=np.linspace(0,10,30)
rng = np.random.default_rng(0)
colors = rng.random(30)
sizes = rng.random(30)*100
plt.plot(x, np.sin(x), '--o', color='g', markersize=10)
plt.plot(x, np.sin(x)+1, '-+', color='b', linewidth=2)
plt.plot(x, np.sin(x)+2, '>', color='k')
plt.plot(x, np.sin(x)+3, 's', color='y', markerfacecolor='white', markeredgecolor='k', markeredgewidth=2)

plt.scatter(x, np.sin(x)+5, marker='o', c=colors, s=sizes)


# points > 1000 -> choose plot

plt.colorbar()
dy=0.4
y = np.sin(x) + dy * np.random.randn(30)
plt.errorbar(x,y,yerr=dy, fmt ='.k')
plt.fill_between(x, y-dy, y+dy, color = 'red', alpha=0.4)



def f(x,y):
    return np.sin(x) ** 5 + np.cos(20+x+y)*np.cos(x)

x = np.linspace(0,5,50)
y = np.linspace(0,5,40)

X, Y = np.meshgrid(x,y)

Z = f(X, Y)

#plt.contour(X, Y, Z, cmap='RdGy')
#plt.contourf(X, Y, Z, cmap='RdGy')

c = plt.contour(X, Y, Z)
plt.clabel(c)
plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian', origin='lower')
plt.colorbar()
