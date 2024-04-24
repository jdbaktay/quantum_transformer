import numpy as np
import scipy.linalg as spla

def calc_dQdK(aI, zII, zIJ, xI, xlist, Q, K):
    Qterm1 = -aI * np.outer(K @ xI, xI)
    Kterm1 = -aI * np.outer(Q @ xI, xI)

    Qterm2, Kterm2 = 0, 0
    for J, row in enumerate(xlist):
        Qterm2 += aI**2 * np.exp(-zIJ[J] + zII) * np.outer(K @ row, xI)
        Kterm2 += aI**2 * np.exp(-zIJ[J] + zII) * np.outer(Q @ xI, row)

    Qder = (Qterm1 + Qterm2) * (1 / np.sqrt(L))
    Kder = (Kterm1 + Kterm2) * (1 / np.sqrt(L))
    return Qder, Kder

def calc_aI(state, Q, K, V, W):
    xlist = state.reshape(Nc, L)

    Qx = np.matmul(xlist, Q.T)
    Kx = np.matmul(xlist, K.T)
    Vx = np.matmul(xlist, V.T)

    z = np.matmul(Qx, Kx.T) / np.sqrt(L) 

    alist = np.zeros(Nc, dtype=np.cfloat)
    for I in range(Nc):
        num = np.exp(-z[I,I])

        denom = 0
        for J in range(Nc):
            denom += np.exp(-z[I,J])

        alist[I] = num / denom

    vtilde = (alist[:, np.newaxis] * Vx).reshape(N)
    coeff = vtilde.T @ W @ vtilde 
    return alist, z, coeff

N = 12
L = 2
Nc = N // L

W = np.random.rand(N, N) + 1j * np.random.rand(N, N)
Q = np.random.rand(L, L) + 1j * np.random.rand(L, L)
K = np.random.rand(L, L) + 1j * np.random.rand(L, L)
V = np.random.rand(L, L) + 1j * np.random.rand(L, L)

state = np.ones(N)
state[:N//2] = -1
state *= 0.5
state = state[np.random.permutation(N)]

aI, z, coeff = calc_aI(state, Q, K, V, W)

xlist = state.reshape(Nc, L)

Vx = np.matmul(xlist, V.T)

daI_dQ, daI_dK = [], []
for I in range(Nc):
    zII = z[I, I]
    zIJ = z[I, :]
    xI = xlist[I]

    Qder, Kder = calc_dQdK(aI[I], zII, zIJ, xI, xlist, Q, K)

    daI_dQ.append(Qder)
    daI_dK.append(Kder)

daI_dQ = np.array(daI_dQ)
daI_dK = np.array(daI_dK)

daI_dQ_v = np.zeros((Nc, L, L, L), dtype=complex)
daI_dK_v = np.zeros((Nc, L, L, L), dtype=complex)
for I in range(Nc):
    for J in range(L):
        daI_dQ_v[I, J, :, :] = Vx[I, J] * daI_dQ[I, :, :]
        daI_dK_v[I, J, :, :] = Vx[I, J] * daI_dK[I, :, :]

daI_dQ_v = daI_dQ_v.reshape(Nc * L, L, L)
daI_dK_v = daI_dK_v.reshape(Nc * L, L, L)

aJv = (aI[:, np.newaxis] * Vx).reshape(N)

Qder1 = np.tensordot(daI_dQ_v, W @ aJv, axes=(0, 0))
Qder2 = np.tensordot(W, daI_dQ_v, axes=(1, 0))
Qder2 = np.tensordot(aJv, Qder2, axes=(0, 0))
Qder = Qder1 + Qder2

Kder1 = np.tensordot(daI_dK_v, W @ aJv, axes=(0, 0))
Kder2 = np.tensordot(W, daI_dK_v, axes=(1, 0))
Kder2 = np.tensordot(aJv, Kder2, axes=(0, 0))
Kder = Kder1 + Kder2

manDerQ = np.zeros((L, L), dtype=complex)
manDerK = np.zeros((L, L), dtype=complex)

D = 1e-4 + 1e-4j
for k in range(L):
    for l in range(L):
        Q1 = Q.copy()
        K1 = K.copy()

        Q1[k, l] += D
        K1[k, l] += D

        coeffQ = calc_aI(state, Q1, K, V, W)[-1]
        coeffK = calc_aI(state, Q, K1, V, W)[-1]

        manDerQ[k, l] = (coeffQ - coeff) / D
        manDerK[k, l] = (coeffK - coeff) / D


print(spla.norm(manDerQ - Qder))
print(spla.norm(manDerK - Kder))
















    









