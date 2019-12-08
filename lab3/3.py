import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from neupy import algorithms
from neupy.layers import Input, Tanh, Linear 

h = 0.01
t = np.linspace(0, 3.5, int(3.5/0.02), endpoint=True)
x = np.cos(np.cos(t) * t**2 + 5*t)

train_size = int(t.shape[0] * 0.9)
train_size

X_train = t[:train_size]
y_train = x[:train_size]

X_test = t[train_size:]
y_test = x[train_size:]

scaler_x = StandardScaler()
scaler_y = StandardScaler()
tmp_train_scaled_x = scaler_x.fit_transform(X_train[:, np.newaxis])
tmp_test_scaled_x = scaler_x.transform(X_test[:, np.newaxis])
tmp_train_scaled_y = scaler_y.fit_transform(y_train[:, np.newaxis])

lmnet = algorithms.LevenbergMarquardt((Input(1), Tanh(60), Linear(1)), verbose=True)

lmnet.train(X_train, y_train, epochs=100)


pred_x = lmnet.predict(X_train)
mse = sklearn.metrics.mean_squared_error(y_train, pred_x.flatten())
print(f'RMSE = {np.sqrt(mse)}')

plt.plot(X_train, y_train, label='train')
plt.plot(X_train, pred_x, label='predict')
plt.legend();



pred_x = lmnet.predict(X_test)
mse = sklearn.metrics.mean_squared_error(y_test, pred_x.flatten())
print(f'RMSE = {np.sqrt(mse)}')

plt.plot(X_test, y_test, label='test')
plt.plot(X_test, pred_x, label='predict')
plt.legend();


plt.show()
