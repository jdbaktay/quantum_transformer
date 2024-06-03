import numpy as np
import scipy.linalg as spla
import itertools

def get_logders(state, Q, K, V, W):
    if np.iscomplex(W).any():
        COMPLEX = 1
    else:
        COMPLEX = 0
    dtype = 'float64'
    if COMPLEX:
        dtype = 'complex128'
    # t1= time.time()
    N = len(W[:,0])
    L = len(Q[:,0])
    Nc = N // L
    xlist = np.reshape(state,(Nc,L))
    z = np.zeros((Nc,Nc), dtype=dtype)
    for i in range(Nc):
        for j in range(Nc):
            xI = xlist[i]
            xJ = xlist[j]
            qI = np.matmul(Q,xI)
            kJ = np.matmul(K,xJ)
            # z[i,j] = np.inner(np.conj(qI), kJ)/np.sqrt(L)
            z[i,j] = np.inner(qI, kJ)/np.sqrt(L)
            
    a = np.zeros(Nc, dtype=dtype)
    for i in range(Nc):
        aI = np.exp(-z[i,i])
        denom = 0
        for j in range(Nc):
            denom += np.exp(-z[i,j])
            # print(denom)
        aI /= denom
        a[i] = aI
    alist = a
    aVx = []
    ax = []
    Vx = []
    for i in range(Nc):
        aI = alist[i]
        xI = xlist[i] 
        ax.append(aI*xI)
        aVx.append(aI*np.matmul(V,xI))
        Vx.append(np.matmul(V,xI))
    vtilde = np.concatenate(aVx) # vtilde = aVx
    
    OV = np.zeros((L,L), dtype=dtype)
    OQ = np.zeros((L,L), dtype=dtype)
    OK = np.zeros((L,L), dtype=dtype)
    # alist = aa
    for i in range(L):
        for j in range(L):
            for I in range(Nc):
                for J in range(Nc):
                    aI = alist[I]
                    aJ = alist[J]
                    xI = xlist[I]
                    xJ = xlist[J]
                    vI = V@xI
                    vJ = V@xJ
                    for k in range(L):
                        OV[i,j] += aI*xI[j]*W[int(I*L+i), int(J*L+k)]*aJ*vJ[k] \
                            + aI*vI[k]*W[int(I*L+k),int(J*L+i)]*aJ*xJ[j]

    for i in range(L):
        for j in range(L):
            daI_dQ_lst = np.zeros(Nc, dtype=dtype)
            daI_dK_lst = np.zeros(Nc, dtype=dtype)
            for I in range(Nc):
                aI = alist[I]
                xI = xlist[I]
                qI = Q@xI
                kI = K@xI
                daI_dQ_lst[I] = -aI * xI[j]*kI[i]/np.sqrt(L)
                daI_dK_lst[I] = -aI * xI[j]*qI[i]/np.sqrt(L)
                for J in range(Nc):
                    kJ = K@xlist[J]
                    qJ = Q@xlist[J]
                    xJ = xlist[J]
                    daI_dQ_lst[I] += (aI)**2 * np.exp(-z[I,J] + z[I,I]) * xI[j]*kJ[i] / np.sqrt(L)
                    daI_dK_lst[I] += (aI)**2 * np.exp(-z[I,J] + z[I,I]) * xJ[j]*qI[i] / np.sqrt(L)
                    
            for I in range(Nc):
                for J in range(Nc):
                    for k in range(L):
                        for l in range(L):
                            vI = V@xlist[I]
                            vJ = V@xlist[J]
                            aI = alist[I]
                            aJ = alist[J]
                            daIdQ = daI_dQ_lst[I]
                            daJdQ = daI_dQ_lst[J]
                            daIdK = daI_dK_lst[I]
                            daJdK = daI_dK_lst[J]
                            OQ[i,j] += W[int(I*L)+k,int(J*L+l)]*(daIdQ * aJ + \
                                                                  aI * daJdQ) * vI[k]*vJ[l]
                            OK[i,j] += W[int(I*L)+k,int(J*L+l)]*(daIdK* aJ + \
                                                                  aI* daJdK) * vI[k]*vJ[l]

    logderW = np.outer(vtilde, vtilde)
    logderV = OV
    logderQ = OQ
    logderK = OK
    return logderQ, logderK, logderV, logderW

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

def calc_dV(aI, xlist, V, W):
    aJv = (aI[:, np.newaxis] * np.matmul(xlist, V.T)).reshape(N)

    reorder = [i*L + j for j, i in itertools.product(range(L), range(Nc))]

    aIxI = (aI[:, np.newaxis] * xlist).reshape(N)[reorder]

    WaJv = (W @ aJv)[reorder]
    aJvW = (aJv @ W)[reorder]

    WaJv_sets = [WaJv[i * (Nc):(i + 1) * (N // L)] for i in range(L)]
    aJvW_sets = [aJvW[i * (Nc):(i + 1) * (N // L)] for i in range(L)]
    aIxI_sets = [aIxI[i * (Nc):(i + 1) * (N // L)] for i in range(L)]

    logder_V = np.zeros((L, L), dtype=complex)
    for i in range(L):
        for j in range(L):
            logder_V[i, j] = aIxI_sets[j].dot(WaJv_sets[i]) \
                           + aJvW_sets[i].dot(aIxI_sets[j])
    return logder_V

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
L = 4
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

Vder = calc_dV(aI, xlist, V, W)

manDerQ = np.zeros((L, L), dtype=complex)
manDerK = np.zeros((L, L), dtype=complex)
manDerV = np.zeros((L, L), dtype=complex)
manDerW = np.zeros((N, N), dtype=complex)

D = 1e-5 + 1e-5j
for k in range(L):
    for l in range(L):
        Q1 = Q.copy()
        K1 = K.copy()
        V1 = V.copy()
        W1 = W.copy()

        Q1[k, l] += D
        K1[k, l] += D
        V1[k, l] += D
        W1[k, l] += D

        coeffQ = calc_aI(state, Q1, K, V, W)[-1]
        coeffK = calc_aI(state, Q, K1, V, W)[-1]
        coeffV = calc_aI(state, Q, K, V1, W)[-1]
        coeffW = calc_aI(state, Q, K, V, W1)[-1]

        manDerQ[k, l] = (coeffQ - coeff) / D
        manDerK[k, l] = (coeffK - coeff) / D
        manDerV[k, l] = (coeffV - coeff) / D
        manDerW[k, l] = (coeffW - coeff) / D

print()
print(spla.norm(manDerQ - Qder))
print(spla.norm(manDerK - Kder))
print(spla.norm(manDerV - Vder))













    









