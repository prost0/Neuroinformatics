import numpy as np
import matplotlib.pyplot as plt
import sys

samples1 = np.array([[4.3, 2.3, 3.6, 4.8, 2.8, -3.3],
					[2.2, -4.4, 4.3, 3.5, 0.1, -1.1]]).T
samples2 = np.array([[-4.4, 0.2, 1.5, -2.1, -4.9, -3.4, -1.3, -0.2],
					[ -1.1, -0.9, 1.2, -0.7, 4.8, -4, -3.1, -1.7]]).T
answers1 = np.array([1, 0, 1, 1, 1, 0])
answers2 = np.array([[1, 0, 0, 0, 0, 1, 0, 0],
					 [0, 0, 1, 0, 1, 0, 0, 0]]).T


w1 = np.array([0, 0], float)
w2 = np.array([[0, 0], [0, 0]], float)
b1 = 0
b2 = np.array([0, 0], float)
learn_rate = 0.1
max_iterations = 10000


def f(x, w, b):
	return x.dot(w) + b

def train(samples, answers, w, b, learn_rate, flag, max_i):
    i = 0
    while i < samples.T[0].size:
        err = answers[i] - np.heaviside(f(samples[i], w, b), 1)
        if np.linalg.norm(err) != 0:
            if(flag == 1):
                w += learn_rate * samples[i].reshape(-1, 1).dot(err.reshape(1, -1))
            else:
                w += learn_rate * samples[i] * err
            b += learn_rate * err
            i = -1
        i += 1
        max_i -= 1
        if max_i == 0:
            sys.exit("Error! Task is not linearly solvable.")
    return  w, b

def predict(samples, w, b, answers):
    i = 0
    while i < samples.T[0].size:
        answers[i] =  np.heaviside(f(samples[i], w, b), 1)
        i += 1

plt.subplot(211)
#sctr1 = plt.scatter(x=samples1.T[0], y=samples1.T[1], s = 100, c=answers1, cmap=plt.cm.RdYlGn)
print(train(samples1, answers1, w1, b1, learn_rate, 0, max_iterations))
s1 =  np.array([[1, 2, 3, -1, -2, -3], [4, -4, 2, -2, 0, 0]]).T
ans1 =  np.array([0, 0, 0, 0, 0, 0])
predict(s1, w1, b1, ans1)
sctr1 = plt.scatter(x=s1.T[0], y=s1.T[1], s = 100, c=ans1, cmap=plt.cm.RdYlGn)
x1 = np.linspace(-5, 5, 100) 
x2 = (-x1 * w1[0] - b1) / w1[1]
plt.plot(x1, x2, color = 'black')

plt.subplot(212)
#sctr1 = plt.scatter(x=samples2.T[0], y=samples2.T[1], s = 100, c=answers2.T[0], cmap=plt.cm.RdYlGn)
#sctr2 = plt.scatter(x=samples2.T[0], y=samples2.T[1], s = 10, c=answers2.T[1], cmap=plt.cm.RdYlGn)
s2 =  np.array([[1, 1, 2, 2, 3, 4, -4, -3], [4, -4, 2, -2, -3, 3, -4, 3]]).T
ans2 =  np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]).T
print(train(samples2, answers2, w2, b2, learn_rate, 1, max_iterations))
predict(s2, w2, b2, ans2)
sctr1 = plt.scatter(x=s2.T[0], y=s2.T[1], s = 100, c=ans2.T[0], cmap=plt.cm.RdYlGn)
sctr2 = plt.scatter(x=s2.T[0], y=s2.T[1], s = 10, c=ans2.T[1], cmap=plt.cm.RdYlGn)

x1 = np.linspace(-5, 5, 100) 
x2 = (-x1 * w2[0][0] - b2[0]) / w2[1][0]
x4 = (-x1 * w2[0][1] - b2[1]) / w2[1][1]
plt.plot(x1, x2, color = 'black')
plt.plot(x1, x4, color = 'gray')

plt.show()