import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, Adagrad

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


a1 = 0.4
b1 = 0.4
x0_1 = 0.1
y0_1 = -0.15
a2 = 0.7
b2 = 0.7
x0_2 = 0
y0_2 = 0
p3 = -1
x0_3 = 0.8
y0_3 = -0.8


step = 0.025
t = np.linspace(0, 2*np.pi, int(2*np.pi/step), endpoint=True)

x = np.linspace(-1, 1, int(2/step), endpoint=True)

def f(x0, a, t):
    return x0 + a*np.cos(t)

def g(y0, b, t):
    return y0 + b*np.sin(t)


x1 = f(x0_1, a1, t)
y1 = g(y0_1, b1, t)
x2= f(x0_2, a2, t)
y2= g(y0_2, b2, t)
#x3= f(a3, t)
#y3= g(b3, t)
x3 = x + x0_3
y3 = p3 * x * x + y0_3

plt.plot(x1, y1, 'navy')
plt.plot(x2, y2, 'purple')
plt.plot(x3, y3, 'yellow')
plt.grid(True)
plt.show()


df1 = pd.DataFrame({'x' : x1, 'y' : y1, 'target' : 0})
df2 = pd.DataFrame({'x' : x2, 'y' : y2, 'target' : 1})
df3 = pd.DataFrame({'x' : x3, 'y' : y3, 'target' : 2})


def split_df(df):
    x_train, x_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=21)
    x_valid, x_test = train_test_split(x_test, test_size=0.3, shuffle=True, random_state=14)
    return x_train, x_valid, x_test


model = Sequential()
model.add(Dense(20, input_shape=(2,), activation='tanh'))
model.add(Dense(3, activation='sigmoid'))

model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])


train = []
valid = []
test = []

for df in (df1, df2 ,df3):
    tr, v, te = split_df(df)
    train.append(tr)
    valid.append(v)
    test.append(te)
train = pd.concat(train)
valid = pd.concat(valid)
test = pd.concat(test)


y = pd.get_dummies(train['target'])
history = model.fit(train.iloc[:, :-1], y, epochs=300, shuffle=True)



p = []

p.append(model.predict_classes(train.iloc[:, :-1]))
accuracy_score(train['target'], p[-1])


p.append(model.predict_classes(test.iloc[:, :-1]))
accuracy_score(test['target'], p[-1])


p.append(model.predict_classes(valid.iloc[:, :-1]))
accuracy_score(valid['target'], p[-1])


titles = ['train', 'test', 'valid']

for idx, df in enumerate((train, test, valid)):
    plt.scatter(df.x, df.y, c=p[idx], cmap=plt.cm.plasma)
    plt.grid(True)
    plt.title(titles[idx])
    plt.show()

    
h = 0.025
grid_pred = [model.predict(np.array([[i, j]])).round(1) for i in np.arange(-1.2, 1.2+h, h)
                                                       for j in np.arange(-1.2, 1.2+h, h)]


x_vals = np.arange(-1.2, 1.2+h, h)
y_vals = np.arange(-1.2, 1.2+h, h)

xx, yy = np.meshgrid(x_vals, y_vals)


rows = len(grid_pred)
colors = np.array(grid_pred).reshape((rows, 3))
colors.shape


plt.scatter(yy, xx, c=colors, cmap=plt.cm.plasma);
plt.show()