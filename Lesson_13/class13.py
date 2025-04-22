#Метод опорных векторов - SVM
#Разделяющая классификация
#Выбирается линия, разделяющая данные с максимальным отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[data['species'] != "virginica"]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_setosa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])

plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

plt.show()
from sklearn.svm import SVC

model = SVC(kernel='linear', C = 1000)

model.fit(X, Y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolor='none', edgecolors='k')

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


X_p_setosa = X_p[X_p['species']=='setosa']
X_p_versicolor = X_p[X_p['species']!='setosa']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)



plt.show()

data = iris[['sepal_length', 'petal_length', 'species']]
# В случае перекрытия данных, то идеальной границы не существует. У модели имеет гиперпараметр, определяющий 'размытие'
# Если C большое, то отступ "жесткий", чем меньше C, тем отступ становитсяболее размытым
print(data.shape)
data_df = data[data['species'] != "setosa"]
print(data_df.shape)

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

c_value = [[10000, 1000, 100, 10], [1, 0.1, 0.01, 0.001]]

fig, ax = plt.subplots(2, 4, sharex = 'col', sharey='row', figsize = (16,10))

x1_p = np.linspace(min(data_df['sepal_length']),max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']),max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)



data_df_setosa = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

X_p['species'] =0.0

for i in range(2):
    for j in range(4):
        ax[i,j].scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])
        ax[i,j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

        model = SVC(kernel='linear', C=c_value[i][j])
        model.fit(X, Y)



        ax[i,j].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolor='none', edgecolors='k')

        X_p =pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, 
            columns=['sepal_length', 'petal_length']
        )

        y_p = model.predict(X_p)


        X_p['species'] = y_p


        X_p_setosa = X_p[X_p['species']=='virginica']
        X_p_versicolor = X_p[X_p['species']!='virginica']
        ax[i,j].scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1, color='green')
        ax[i,j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1, color='red')
plt.show()

# Достоинства:
# - Зависимость от небольшого числа опорных векторов => компактность модели
# - После обучения предсказания проходять очень быстро
# - на работу метода влияют только точки, находящиеся возле отступов, поэтому методы подходят для многомерных данных

# Недостатки
# - При большой выборке могут быть значительные вычислительные затраты
# - Большая зависимость от размытости C. Поиск может привести к большим вычислительным затратам
# - У результатов отсутствует вероятностная интерпритация 


# Деревья решений и случайный лес
# Случайный лес непараметрический алгоритм
# СЛ - пример ансамблиевого метода, основанного на агрегации результатов множества простых моделей
# В реализациях дерева принятия решений в машинном обучении, вопросы обычно ведут к разделению данных по осям
# Каждый узел дерева разбивает данные на две группы по одному из признаков

from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')

species_int = []

for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)


species_int_df = pd.DataFrame(species_int)


data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int_df

data_df = data[data['species'] != 1]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_setosa = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]

plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])

plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])


model=DecisionTreeClassifier(max_depth=4)

model.fit(X, Y)



x1_p = np.linspace(min(data_df['sepal_length']),max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']),max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)


X_p =pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, 
    columns=['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)


#plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1)
#plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)

plt.contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1)
plt.colorbar()
plt.show()
