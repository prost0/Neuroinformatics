import numpy as np
import matplotlib.pyplot as plt
import sys
import math


def out2F(t):
    return (1/4) * math.cos(-math.cos(t) * t**2 + t + 2*math.pi)

def in1F(t):
    return math.sin(-2*math.sin(t) * t**2 + 7*t)

def in2F(t):
    return math.cos(math.cos(t)*t**2 + t)

out2F = np.vectorize(out2F)
in1F = np.vectorize(in1F)
in2F = np.vectorize(in2F)

'''
in2 = (np.linspace(0.5, 4.0, num = (4.0 - 0.5) / 0.01))
out2 = out2F(in2)
in2 = in2F(in2)
'''
'''
ls1_x = np.linspace(0.5, 3.2, num = 1 + (3.2 - 0.5) / 0.01)
ls1 = in1F(ls1_x)
#out1 = in1[3::4]
#in1 = np.delete(in1, np.arange(3, in1.size, 4))
w1 = np.array([1, 1, 1, 1, 1], float)
b1 = 0
learn_rate1 = 0.01
number_of_era1 = 5
e1 = 99999
out1 = np.zeros(ls1.size)

for i in range(50):
    e1 = 0
    for i in range(ls1.size - number_of_era1):
        x1 = np.array([ls1[i], ls1[i + 1], ls1[i + 2], ls1[i + 3], ls1[i + 4]], float)
        out1[i + number_of_era1] = w1.dot(x1) + b1
        w1 -= learn_rate1 * (out1[i + number_of_era1] - ls1[i + number_of_era1]) * x1
        b1 -= learn_rate1 * (out1[i + number_of_era1] - ls1[i + number_of_era1])
        e1 += ((out1[i + number_of_era1] - ls1[i + number_of_era1]) ** 2) / 2
        
print("e1 = ", e1)
plt.subplot(311)
plt.plot(ls1_x, ls1, color = 'red')
plt.plot(ls1_x, out1, color = 'blue')

###################################################################

ls2_x = np.linspace(0.5, 3.21, num = 11 + (3.2 - 0.5) / 0.01)
ls2 = in1F(ls2_x)

w2 = np.array([1, 1, 1], float)
b2 = 0
learn_rate2 = 0.01
number_of_era2 = 3
e2_m = 10**(-6)
e2 = 99999
out2 = np.zeros(ls2.size)

#while e1 > e1_m:
for i in range(600):
    if e2 < e2_m:
        for i in range(ls2.size - number_of_era2):
            x2 = np.array([ls2[i], ls2[i + 1], ls2[i + 2]], float)
            out2[i + 3] = w2.dot(x2) + b2
        break
    else: 
        e2 = 0
        for i in range(ls2.size - number_of_era2):
            x2 = np.array([ls2[i], ls2[i + 1], ls2[i + 2]], float)
            out2[i + 3] = w2.dot(x2) + b2
            w2 -= learn_rate2 * (out2[i + number_of_era2] - ls2[i + number_of_era2]) * x2
            b2 -= learn_rate2 * (out2[i + number_of_era2] - ls2[i + number_of_era2])
            e2 += ((out2[i + number_of_era2] - ls2[i + number_of_era2]) ** 2) / 2

print("e2 = " , e2)
plt.subplot(312)
plt.plot(ls2_x, ls2, color = 'red')
plt.plot(ls2_x, out2, color = 'blue')
'''
##########################################################################

ls3_x = np.linspace(0.5, 4., num = 1 + (4. - 0.5) / 0.01)
ls3 = in2F(ls3_x)
ls3_out = out2F(ls3_x)

w3 = np.array([0, 0, 0, 0, 0, 0, 0], float)
b3 = 0
learn_rate3 = 0.001
number_of_era3 = 7
e3_m = 10**(-6)
e3 = 99999
out3 = np.zeros(ls3.size)

for i in range(1000):

    e3 = 0
    for i in range(ls3.size - number_of_era3):
        x3 = np.array([ls3[i], ls3[i + 1], ls3[i + 2], ls3[i + 3], ls3[i+4], ls3[i+5], ls3[i +6]], float)
        out3[i + number_of_era3] = w3.dot(x3) + b3
        w3 -= learn_rate3 * (out3[i + number_of_era3] - ls3_out[i + number_of_era3]) * x3
        b3 -= learn_rate3 * (out3[i + number_of_era3] - ls3_out[i + number_of_era3])
        e3 += ((out3[i + number_of_era3] - ls3_out[i + number_of_era3]) ** 2) / 2

print("e3 = " , e3)
plt.subplot(313)
plt.plot(ls3_x, ls3_out, color = 'red')
plt.plot(ls3_x, out3, color = 'blue')
plt.plot(ls3_x, ls3, color = 'green')

plt.show()

