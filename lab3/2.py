import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, Adagrad

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


h = 0.01
t = np.linspace(0, 3.5, int(3.5/0.02), endpoint=True)
x = np.cos(np.cos(t) * t**2 + 5*t)


plt.plot(t, x);


train_size = int(t.shape[0] * 0.9)


X_train = t[:train_size]
y_train = x[:train_size]

X_test = t[train_size:]
y_test = x[train_size:]


scaler_x = StandardScaler()
scaler_y = StandardScaler()
tmp_train_scaled_x = scaler_x.fit_transform(X_train[:, np.newaxis])
tmp_test_scaled_x = scaler_x.transform(X_test[:, np.newaxis])
tmp_train_scaled_y = scaler_y.fit_transform(y_train[:, np.newaxis])

model = Sequential()
model.add(Dense(20, input_shape=(1,), activation='tanh'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(tmp_train_scaled_x, tmp_train_scaled_y , epochs=12000, verbose=1)


pred_x = model.predict(tmp_train_scaled_x)
pred_x = scaler_y.inverse_transform(pred_x)
mse = mean_squared_error(y_train, pred_x.flatten())
print(f'RMSE = {np.sqrt(mse)}')

plt.plot(X_train, y_train, label='train')
plt.plot(X_train, pred_x, label='predict')
plt.legend();


pred_x = model.predict(tmp_test_scaled_x[:, 0])
pred_x = scaler_y.inverse_transform(pred_x)
mse = mean_squared_error(y_test, pred_x.flatten())
print(f'RMSE = {np.sqrt(mse)}')

plt.plot(X_test, y_test, label='test')
plt.plot(X_test, pred_x, label='predict')
plt.legend();
plt.show()