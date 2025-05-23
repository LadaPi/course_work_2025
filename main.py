import numpy as np
import matplotlib.pyplot as plt
import random
from algorithm import oper_extrapolation
from matplotlib.lines import Line2D

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

fig, axs = plt.subplots(3, 1, figsize=(10, 7))

x_sin = np.sin(2*t)
sigma = 0.3*np.std(x_sin)
x_sin_noisy = x_sin + sigma*np.random.randn(n)
x_sin_denoised = oper_extrapolation(x_sin_noisy, 1.5, 1e-10)[0]

axs[0].plot(t, x_sin, color="blue")
axs[0].plot(t, x_sin_noisy, lw=0.9, color="green")
axs[0].plot(t, x_sin_denoised, lw=1.2, color="red")

x_p_const = p_const(t)
sigma = 0.3*np.std(x_p_const)
x_p_const_noisy = x_p_const + sigma*np.random.randn(n)
x_p_const_denoised = oper_extrapolation(x_p_const_noisy, 3, 1e-10)[0]

axs[1].plot(t, x_p_const, color="blue")
axs[1].plot(t, x_p_const_noisy, lw=0.9, color="green")
axs[1].plot(t, x_p_const_denoised, lw=1.2, color="red")

x_rand = rand_sygnal(t)
sigma = 0.3*np.std(x_rand)
x_rand_noisy = x_rand + sigma*np.random.randn(n)
x_rand_denoised = oper_extrapolation(x_rand_noisy, 100, 1e-10)[0]

axs[2].plot(t, x_rand, color="blue")
axs[2].plot(t, x_rand_noisy, lw=0.9, color="green")
axs[2].plot(t, x_rand_denoised, lw=1.2, color="red")

legend_elements = [
    Line2D([0], [0], color='blue', label='Початковий сигнал'),
    Line2D([0], [0], color='green', label='Зашумлений сигнал'),
    Line2D([0], [0], color='red', label='Знешумлений сигнал')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False)

plt.tight_layout(rect=[0.02, 0.05, 0.95, 1])
plt.show()
