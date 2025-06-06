import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from fastai.text.all import *

# ## Обнаружение аномалий
# Поиск аномальных транзакций (мошеннеческих).
# 
# Просматриваются образцы данных, и среди них находятся аномальные.
# 
# Будем использовать метод главных компонент:
# - уменьшим размерность данных
# - восстановим размерность данных
# 
# Для аномальных данных потеря составит больше, чем для нормальных данных.

df = pd.read_csv('./data/creditcard.csv')
# print(df.head())

legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

legit = legit.drop(['Class', 'Time'], axis=1)
fraud = fraud.drop(['Class', 'Time'], axis=1)

# print(df.shape, legit.shape, fraud.shape)


# **Уменьшение размерности**

pca = PCA(n_components=26, random_state=0)
# Данные с уменьшенной размерностью
legit_pca = pd.DataFrame(pca.fit_transform(legit), index=legit.index)
fraud_pca = pd.DataFrame(pca.transform(fraud), index=fraud.index)
print(legit_pca.shape, fraud_pca.shape)


# **Восстановление данных**

legit_restore = pd.DataFrame(pca.inverse_transform(legit_pca), index=legit_pca.index)
fraud_restore = pd.DataFrame(pca.inverse_transform(fraud_pca), index=fraud_pca.index)
print(legit_restore.shape, fraud_restore.shape)


# **Поиск разности**


def anomaly_calc(original, restored):
    loss = np.sum((np.array(original) - np.array(restored)) ** 2, axis=1)
    return pd.Series(data=loss, index=original.index)

legit_calc = anomaly_calc(legit, legit_restore)
fraud_calc = anomaly_calc(fraud, fraud_restore)


fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
ax[0].plot(legit_calc)
ax[1].plot(fraud_calc)


th = 1
legit_TRUE = legit_calc[legit_calc < th].count()
legit_FALSE = legit_calc[legit_calc >= th].count()

fraud_TRUE = fraud_calc[fraud_calc >= th].count()
fraud_FALSE = fraud_calc[fraud_calc < th].count()

print(legit_TRUE, legit_FALSE, fraud_TRUE, fraud_FALSE)


# ## Рекуррентные нейронные сети
# ### Обработка естественного языка
# В основе лежит **языковая модель**, которая позволяет предсказывать следующее слово, зная предыдущие.
# 
# Метки не требуются, но нужно очень много текста. Они получаются автоматически из данных.

path = untar_data(URLs.HUMAN_NUMBERS)

print(path.ls())


lines = L()
with open('C:/Users/user/.fastai/data/human_numbers/data.txt') as f:
    lines += L(*f.readlines())

# print(lines[:10])
text = ' '.join([l.strip() for l in lines])
# print(text[:50])
tokens = text.split(' ')
# print(tokens[:10])
vocab = L(*tokens).unique()
#print(vocab)
word2index = {w: i for i,w in enumerate(vocab)}
# print(word2index)

nums = L(word2index[i] for i in tokens)

seq = L((tokens[i:i+3], tokens[i+3]) for i in range(0, len(tokens)-4, 3))
print(seq[:10])

seq = L((nums[i:i+3], nums[i+3]) for i in range(0, len(nums)-4, 3))
print(seq[:10])

seq = L((tensor(nums[i:i+3]), nums[i+3]) for i in range(0, len(nums)-4, 3))
print(seq[:10])

bs = 64
cut = int(len(seq) * 0.8)
dls = DataLoaders.from_dsets(seq[:cut], seq[cut:], bs=bs, shuffle=False)


# #### Обучение
# Будут входной слой, 3 скрытых слоя (1, 2 и 3) и выходной слой. Сеть является полносвязной.
# 
# На вход подаются батчи, состоящие из последовательности слов.
# 
# Первое слово идет на первый слой. 
# Второе слово идет на второй слой, перед чем суммируется с результатом функции активации от первого слова.
# Третье слово идет на третий слой, перед чем суммируется с результатом функции от предыдущего шага.
# Смещения и веса для всех слоев одинаковы.


class Model1(Module):
	def __init__(self, vocab_sz, n_hidden):
		self.i_h = nn.Embedding(vocab_sz, n_hidden)
		self.h_h = nn.Linear(n_hidden, n_hidden)
		self.h_o = nn.Linear(n_hidden, vocab_sz)

	def forward(self, x):
		h = F.relu(self.h_h(self.i_h(x[:, 0])))
		h = h + self.i_h(x[:, 1])
		h = F.relu(self.h_h(h)) # h2
		h = h+ self.i_h(x[:, 2])
		h = F.relu(self.h_h(h)) 
		return self.h_o(h)


learn = Learner(dls, Model1(len(vocab), bs), loss_func=F.cross_entropy, metrics=accuracy)

learn.fit_one_cycle(4, 0.001)


n = 0
counts = torch.zeros(len(vocab))

for x,y in dls.valid:
	n += y.shape[0]
	for i in range_of(vocab):
		counts[i] += (y == i).long().sum()
print(counts)
index = torch.argmax(counts)
print(index, vocab[index.item()], counts[index].item()/n)


# **Рекуррентная нейронная сеть**, определенная с помощью цикла


class Model2(Module):
	def __init__(self, vocab_sz, n_hidden):
		self.i_h = nn.Embedding(vocab_sz, n_hidden)
		self.h_h = nn.Linear(n_hidden, n_hidden)
		self.h_o = nn.Linear(n_hidden, vocab_sz)

	def forward(self, x):
		h = 0
		for i in range(3):
			h = h + self.i_h(x[:, i])
			h = F.relu(self.h_h(h))
		return self.h_o(h)

learn = Learner(dls, Model2(len(vocab), bs), loss_func=F.cross_entropy, metrics=accuracy)

learn.fit_one_cycle(4, 0.001)