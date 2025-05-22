import numpy as np
Lambda = 2 # коригуючий параметр

def norm_inf(x):
    return max([abs(xi) for xi in x])

def norm_euc(x):
    n = len(x)
    S = 0
    for i in range(n):
        S+=x[i]*x[i]
    return np.sqrt(S)

def proj_on_ball(x, Lambda):
    norm_x = norm_inf(x)
    if (norm_x <= Lambda):
        return x
    else:
        return Lambda/norm_x * np.array(x)

def create_D(n):
    D = np.zeros(shape=(n-1, n))
    for i in range(n-1):
        for j in range(n):
            if i == j:
                D[i,j] = 1
            elif j == i+1:
                D[i, j] = -1
    return D

def lipschitz_const(D):
    evc_norm = np.linalg.norm(D, ord=2)
    return np.sqrt(max(2+evc_norm**2, 2*evc_norm**2))

def oper_extrapolation(y, Lambda, eps):
    d = len(y)
    D = create_D(d)
    L = lipschitz_const(D)

    xn = np.zeros(d) # x0
    xn_1 = np.zeros(d) # x1
    pn = np.zeros(d-1) # p0
    pn_1 = np.zeros(d-1) # p1

    nabla1c = xn_1 - y + np.dot(D.T,pn_1)
    nabla1p = xn - y + np.dot(D.T,pn)
    nabla2c = np.dot(D,xn_1)
    nabla2p = np.dot(D,xn)
    xn_2 = xn_1 - 1.8 / (2 * L) * np.array(nabla1c) + 0.9 / (2 * L) * np.array(nabla1p) # це щоб умову перевірити
    pn_2 = proj_on_ball(pn_1 + 1.8/(2*L) * np.array(nabla2c) - 0.9/(2*L) * np.array(nabla2p), Lambda)
    while not (
            norm_euc(xn - xn_1) < eps and norm_euc(xn_1 - xn_2) < eps and
            norm_inf(pn - pn_1) < eps and norm_inf(pn_1 - pn_2) < eps
    ):
        xn = xn_1; xn_1 = xn_2
        pn = pn_1; pn_1 = pn_2

        nabla1p = nabla1c
        nabla1c = xn_1 - y + np.dot(D.T,pn_1)
        nabla2p = nabla2c
        nabla2c = np.dot(D,xn_1)
        xn_2 = xn_1 - 1.8 / (2 * L) * np.array(nabla1c) + 0.9 / (2 * L) * np.array(nabla1p)
        pn_2 = proj_on_ball(pn_1 + 1.8 / (2 * L) * np.array(nabla2c) - 0.9 / (2 * L) * np.array(nabla2p), Lambda)

    return xn_1, pn_1

