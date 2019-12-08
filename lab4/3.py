import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neupy.algorithms import PNN, GRNN
from neupy import estimators

h = 0.01
t = np.linspace(0, 3.5, int(3.5/0.02), endpoint=True)
x = np.cos(np.cos(t) * t**2 + 5*t)
plt.plot(t, x, c = 'r');
plt.show()

def build_model2(std, train_size, t, x):
    train_size = int(t.shape[0] * train_size)

    # X_train = t[:train_size]
    # y_train = x[:train_size]
    # X_test = t[train_size:]
    # y_test = x[train_size:]
    
    X_train, X_test, y_train, y_test = train_test_split(t, x, train_size=train_size, shuffle=True, random_state=14)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    tmp_train_scaled_x = scaler_x.fit_transform(X_train[:, np.newaxis])
    tmp_test_scaled_x = scaler_x.transform(X_test[:, np.newaxis])
    tmp_train_scaled_y = scaler_y.fit_transform(y_train[:, np.newaxis])

    grnn = GRNN(std=std)
    grnn.fit(tmp_train_scaled_x, tmp_train_scaled_y)

    pred_x = grnn.predict(tmp_train_scaled_x)
    pred_x = scaler_y.inverse_transform(pred_x)
    mse = mean_squared_error(y_train, pred_x.flatten())
    print(f'RMSE = {np.sqrt(mse)}')

    plt.plot(t, x, c = 'r');
    plt.scatter(X_train, y_train, label='train')
    plt.scatter(X_train, pred_x, label='predict')
    plt.legend()
    plt.show()

    pred_x = grnn.predict(tmp_test_scaled_x)
    pred_x = scaler_y.inverse_transform(pred_x)
    mse = mean_squared_error(y_test, pred_x.flatten())
    print(f'RMSE = {np.sqrt(mse)}')

    plt.plot(t, x, c = 'r');
    plt.scatter(X_test, y_test, label='test')
    plt.scatter(X_test, pred_x, label='predict')
    plt.legend()
    plt.show()


build_model2(0.05, 0.8, t, x)
