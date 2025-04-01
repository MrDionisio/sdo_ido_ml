# конспект лекции 4
# выполнил: Кучиев Денис Юрьевич

import numpy as np
import pandas as pd

# если размерность данных > 2, используют иерархическую индексацию (мультииндекс)
index = [
    ('city_1',2010, 1),
    ('city_1',2010, 2),
    ('city_1',2020, 1),
    ('city_1',2020, 2),
    ('city_2',2010, 1),
    ('city_2',2010, 2),
    ('city_2',2020, 1),
    ('city_2',2020, 2),
    ('city_3',2010, 1),
    ('city_3',2010, 2),
    ('city_3',2020, 1),
    ('city_3',2020, 2)
]

population = [101,1010,201,2010,102,1020,202,2020,103,1030,203,2030]

pop = pd.Series(population, index=index)
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)

print(pop)  # видна иерархия
print(pop[:,2020])
print(pop[:,:,2])

# конвертация между Series и DataFrame
pop_df = pop.unstack()
print(pop_df)
print(pop_df.stack())

index = [
    ('city_1',2010),
    ('city_1',2020),
    ('city_2',2010),
    ('city_2',2020),
    ('city_3',2010),
    ('city_3',2020)
]

population = [101,1010,201,2010,102,1020]
pop = pd.Series(population, index=index)
index1 = pd.MultiIndex.from_tuples(index)

pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [11,12,13,14,15,16]
    },
    index=index1
)

print(pop_df)
print(pop_df['something'])

pop_df_1 = pop_df.loc[['city_1'],'something']
pop_df_2 = pop_df.loc[['city_1','city_3'],['total','something']]
pop_df_3 = pop_df.loc[['city_1','city_3'],'something']
print(pop_df_1, pop_df_2, pop_df_3, sep='\n')

### создание мультииндексов разными способами
i1 = pd.MultiIndex.from_arrays([['a','a','b','b'], [1,2,1,2]])
i2 = pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
i3 = pd.MultiIndex.from_product([['a','b'], [1,2]])
i4 = pd.MultiIndex(
    levels=[['a','b','c'], [1,2]],
    codes=[[0,0,1,1,2,2], [0,1,0,1,0,1]]
)

print(i1, i2, i3, i4, sep='\n')

# именование уровней индекса
data = {
    ('city_1',2010): 100,
    ('city_1',2020): 200,
    ('city_2',2010): 1001,
    ('city_2',2020): 2001
}

s = pd.Series(data)
s.index.names = ['city','year']
print(s)

# мультииндекс для строк и столбцов
index = pd.MultiIndex.from_product(
    [['city_1','city_2'], [2010,2020]],
    names=['city','year']
)

columns = pd.MultiIndex.from_product(
    [['person_1','person_2','person_3'], ['job_1','job_2']],
    names=['worker','job']
)

rng = np.random.default_rng(1)
data = rng.random((4,6))
data_df = pd.DataFrame(data, index=index, columns=columns)
print(data_df)

### индексация и срезы
data = {
    ('city_1',2010): 100,
    ('city_1',2020): 200,
    ('city_2',2010): 1001,
    ('city_2',2020): 2001,
    ('city_3',2010): 10010,
    ('city_3',2020): 20010
}

s = pd.Series(data)
s.index.names = ['city','year']
print(s['city_1',2010])
print(s['city_1'])
print(s.loc['city_1':'city_2'])
print(s[:, 2010])
print(s[s > 2000])
print(s[['city_1','city_3']])

### перегруппировка мультииндексов
rng = np.random.default_rng(1)
index = pd.MultiIndex.from_product([['a','c','b'], [1,2]])
data = pd.Series(rng.random(6), index=index)
data.index.names = ['char','int']
data = data.sort_index()
print(data['a':'b'])

# изменение порядка уровней при unstack
print(pop.unstack(level=0))
print(pop.unstack(level=1))
print(pop.unstack(level=2))

### конкатенация
# numpy примеры
x, y, z = [1,2,3], [4,5,6], [7,8,9]
print(np.concatenate([x,y,z]))

x, y, z = [[1,2,3]], [[4,5,6]], [[7,8,9]]
print(np.concatenate([x,y,z], axis=0))
print(np.concatenate([x,y,z], axis=1))

# pandas concat
ser1 = pd.Series(['a','b','c'], index=[1,2,3])
ser2 = pd.Series(['d','e','f'], index=[4,5,6])
print(pd.concat([ser1,ser2]))

ser1 = pd.Series(['a','b','c'], index=[1,2,3])
ser2 = pd.Series(['d','e','f'], index=[1,2,6])
print(pd.concat([ser1,ser2], verify_integrity=False))
print(pd.concat([ser1,ser2], ignore_index=True))
print(pd.concat([ser1,ser2], keys=['x','y']))

ser1 = pd.Series(['a','b','c'], index=[1,2,3])
ser2 = pd.Series(['b','c','f'], index=[4,5,6])
print(pd.concat([ser1,ser2], join='outer'))
print(pd.concat([ser1,ser2], join='inner'))