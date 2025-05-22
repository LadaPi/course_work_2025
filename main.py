import numpy as np
import matplotlib.pyplot as plt
import random
from algorithm import oper_extrapolation

def p_const(t):
    p = []
    for i in range(len(t)):
        if np.sin(t[i])>0:
            p.append(1)
        else:
            p.append(-1)
    return p

def rand_sygnal(t):
    len_t = len(t)
    p = np.zeros(len_t)

    k = 0
    while k < len_t:
        max_step = min(len_t/3, len_t-k) # не хочу щоб сильно великі кроки були
        step = random.randint(1, max_step)
        y = round(random.uniform(-50, 50), 1)
        p[k:k+step] = y
        k+=step
    return p

a = -10; b = 10; n = 300
t = np.linspace(a, b, n)
h = (b-a)/n

plt.figure(figsize=(10, 6))

x_sin = np.sin(2*t)
plt.subplot(3, 1, 1)
sigma = 0.3*np.std(x_sin)
x_sin_noisy = x_sin + sigma*np.random.randn(n)
x_sin_denoised = oper_extrapolation(x_sin_noisy, 1.5, 1e-10)[0]

plt.plot(t, x_sin)
plt.plot(t, x_sin_noisy, lw=0.9, color="green")
plt.plot(t, x_sin_denoised, lw=1.2, color="red")

x_p_const = p_const(t)
plt.subplot(3, 1, 2)
sigma = 0.3*np.std(x_p_const)
x_p_const_noisy = x_p_const + sigma*np.random.randn(n)
x_p_const_denoised = oper_extrapolation(x_p_const_noisy, 3, 1e-10)[0]

plt.plot(t, x_p_const)
plt.plot(t, x_p_const_noisy, lw=0.9, color="green")
plt.plot(t, x_p_const_denoised, lw=1.2, color="red")

x_rand = rand_sygnal(t)
plt.subplot(3, 1, 3)
sigma = 0.3*np.std(x_rand)
x_rand_noisy = x_rand + sigma*np.random.randn(n)
x_rand_denoised = oper_extrapolation(x_rand_noisy, 100, 1e-10)[0]

plt.plot(t, x_rand)
plt.plot(t, x_rand_noisy, lw=0.9, color="green")
plt.plot(t, x_rand_denoised, lw=1.2, color="red")

plt.subplots_adjust(hspace=0.5)
plt.show()
