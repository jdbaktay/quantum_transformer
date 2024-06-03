import jax.numpy as jnp
import jax
import time
import math
import numpy as np
import sys
import itertools
from numpy import exp, sqrt
from matplotlib import pyplot as pl
from jax import config
config.update("jax_enable_x64", True) 

t0 = time.time()

def intersection(list1, list2):
    set1 = set(map(tuple, list1))
    set2 = set(map(tuple, list2))

    intersection_set = set1.intersection(set2)

    intersection_list = list(map(list, intersection_set))
    return intersection_list

class SpinConfig:
    def __init__(self, k, L, Q, K, V, W):
        self.k = k
        self.L = L

        self.spin = np.ones(self.L) 
        self.spin[: self.L//2] = -1
        self.spin *= 0.5
        self.spin = self.spin[np.random.permutation(self.L)]

        self.mult = 1
        self.phase = 1
        self.coefficient = self.Psi(self.spin, Q, K, V, W)

    def translations(self, spins):
        translations = [np.roll(spins, shift) for shift in range(self.L)]
        return translations

    def representative(self, spins):
        new_spins = self.translations(spins)
        rep = np.copy(spins)
        mult = 0
        iphase = 0j
        ik = self.k * 2 * np.pi / self.L
        ntras = 0
        for tras in new_spins:
            nequal = 0
            for i in range(self.L):
                if spins[i] == tras[i]:
                    nequal += 1
            if nequal == self.L:
                mult += 1
                iphase += np.exp(ik * ntras)
            ntras += 1
        for tras in new_spins:
            for i in range(self.L):
                if tras[i] < rep[i]:
                    rep = np.copy(tras)
                    break
                elif rep[i] < tras[i]:
                    break
        if abs(iphase.real) < 1.e-10:
            phase = 0
        else:
            phase = mult
        return rep, mult, phase

    def Psi(self, spins, Q, K, V, W):
        state = spins

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
        return jnp.exp(vtilde.T @ W @ vtilde)

    def metropolis_step(self, Q, K, V, W):
        found = False
        rep = np.copy(self.spin)

        mult = 1
        phase = 1
        i = int(np.random.random() * self.L)
        j = int(np.random.random() * self.L)

        spin_translations = self.translations(self.spin)

        while not found:
            # print(found)
            # print(self.spin)
            while self.spin[i] == self.spin[j]:
                i = int(np.random.random() * self.L)
                j = int(np.random.random() * self.L)

            new_spins = np.copy(self.spin)
            new_spins[i] *= -1
            new_spins[j] *= -1

            # print(new_spins)

            new_spin_translations = self.translations(new_spins)

            inter = intersection(spin_translations, new_spin_translations)

            if inter:
                i = int(np.random.random() * self.L)
                j = int(np.random.random() * self.L)
                continue

            rep, mult, phase = self.representative(new_spins)
            # print('phase=', phase)

            # print(rep)
            # print()

            if phase != 0:
                for n in range(self.L):
                    if self.spin[n] != rep[n]:
                        found = True
                        break

                    ####################################################

                    # Modification

                    # else:
                    #     i = int(np.random.random() * self.L)
                    #     j = int(np.random.random() * self.L)

                    ####################################################


        p = (self.Psi(rep, Q, K, V, W) / self.Psi(self.spin, Q, K, V, W)) ** 2
        p /= mult
        
        if p >= 1 or np.random.random() < p:
            self.spin = np.copy(rep)  # Update to the new state
            self.mult = mult
            self.phase = phase
            self.coefficient = self.Psi(self.spin, Q, K, V, W)  # Update coefficient
            nchanges = 1
        else:
            nchanges = 0

        return nchanges

    def get_spin(self):
        return self.spin

    def get_coefficient(self):
        return self.coefficient

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
    return alist, z

def get_logders(state, Q, K, V, W):
    xlist = state.reshape(Nc, L)

    aI, z = calc_aI(state, Q, K, V, W)

    Vx = np.matmul(xlist, V.T)

    aJv = (aI[:, np.newaxis] * Vx).reshape(N)

    #################### logder_Q and K ################################

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

    Qder1 = np.tensordot(daI_dQ_v, W @ aJv, axes=(0, 0))
    Qder2 = np.tensordot(W, daI_dQ_v, axes=(1, 0))
    Qder2 = np.tensordot(aJv, Qder2, axes=(0, 0))
    tmp_logder_Q = Qder1 + Qder2

    Kder1 = np.tensordot(daI_dK_v, W @ aJv, axes=(0, 0))
    Kder2 = np.tensordot(W, daI_dK_v, axes=(1, 0))
    Kder2 = np.tensordot(aJv, Kder2, axes=(0, 0))
    tmp_logder_K = Kder1 + Kder2

    #################### logder_W ######################################

    tmp_logder_W = np.outer(aJv, aJv)

    #################### logder_V ######################################

    reorder = [i*L + j for j, i in itertools.product(range(L), range(Nc))]

    aIxI = (aI[:, np.newaxis] * xlist).reshape(N)[reorder]

    WaJv = (W @ aJv)[reorder]
    aJvW = (aJv @ W)[reorder]

    WaJv_sets = [WaJv[i * (Nc):(i + 1) * (N // L)] for i in range(L)]
    aJvW_sets = [aJvW[i * (Nc):(i + 1) * (N // L)] for i in range(L)]
    aIxI_sets = [aIxI[i * (Nc):(i + 1) * (N // L)] for i in range(L)]

    tmp_logder_V = np.zeros((L, L), dtype=complex)
    for i in range(L):
        for j in range(L):
            tmp_logder_V[i, j] = aIxI_sets[j].dot(WaJv_sets[i]) \
                               + aJvW_sets[i].dot(aIxI_sets[j])
    return tmp_logder_Q, tmp_logder_K, tmp_logder_V, tmp_logder_W

def get_coeff_2(state, Q, K, V, W):
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
    return jnp.exp(vtilde.T @ W @ vtilde)

def get_eL(state, coeff, Q,K,V,W, MARSHALL_SIGN):
    N = len(state)
    E1 = 0
    E2 = 0
    for i in range(N):
        E1 += state[i]*state[(i+1)%N]

        if (state[i]*state[(i+1)%N] < 0):
            state_new = state.copy()
            state_new[i] *= -1
            state_new[(i+1)%N] *= -1
            
            coeff_new = get_coeff_2(state_new, Q,K,V,W)
            
            E2 += coeff_new / coeff

    if MARSHALL_SIGN:
        return E1 - 0.5 * E2
    else:
        return E1 + 0.5 * E2

def get_E_QKVW_MC_SR(Nsample, Q, K, V, W, MARSHALL_SIGN, L2_1, L2_2):
    N = len(W[0,:])
    L = len(Q[0,:])

    energy_sum = 0.0
    
    if np.iscomplex(W).any():
        COMPLEX = 1
    else:
        COMPLEX = 0
        
    dtype = 'float64'
    if COMPLEX:
        dtype = 'complex128'
    
    logder_W_sum = np.zeros((N,N), dtype= dtype)
    HO_W_sum = np.zeros((N,N), dtype= dtype)
    flat_logder_W_sum = np.zeros(N**2, dtype= dtype)
    logder_outer_W_sum = np.zeros((N**2, N**2), dtype= dtype)
    
    logder_V_sum = np.zeros((L,L), dtype= dtype)
    HO_V_sum = np.zeros((L,L), dtype= dtype)
    flat_logder_V_sum = np.zeros(L**2, dtype= dtype)
    logder_outer_V_sum = np.zeros((L**2, L**2), dtype= dtype)
    
    logder_Q_sum = np.zeros((L,L), dtype= dtype)
    HO_Q_sum = np.zeros((L,L), dtype= dtype)
    flat_logder_Q_sum = np.zeros(L**2, dtype= dtype)
    logder_outer_Q_sum = np.zeros((L**2, L**2), dtype= dtype)
    
    logder_K_sum = np.zeros((L,L), dtype= dtype)
    HO_K_sum = np.zeros((L,L), dtype= dtype)
    flat_logder_K_sum = np.zeros(L**2, dtype= dtype)
    logder_outer_K_sum = np.zeros((L**2, L**2), dtype= dtype)

    k = 0
    spin_config = SpinConfig(k, N, Q, K, V, W)

    print(spin_config.get_spin())
        
    for i in range(Nsample):
        # print('i =', i)
        # for j in range(3): # need this loop?
        nchanges = spin_config.metropolis_step(Q, K, V, W)
        state = spin_config.get_spin()
        coeff = spin_config.get_coefficient()
            
        tmp_energy = get_eL(state, coeff, Q, K, V, W, MARSHALL_SIGN)
             
        tmp_logder_Q, tmp_logder_K, tmp_logder_V, tmp_logder_W \
                                            = get_logders(state, Q, K, V, W)

        energy_sum += tmp_energy 
        
        flat_logder_W_sum += tmp_logder_W.flatten() 
        logder_outer_W_sum += \
            np.outer(np.conj(tmp_logder_W.flatten()), tmp_logder_W.flatten()) 
        tmp_logder_W = np.conj(tmp_logder_W)
        logder_W_sum += tmp_logder_W 
        HO_W_sum += tmp_logder_W * tmp_energy 
        
        flat_logder_V_sum += tmp_logder_V.flatten() 
        logder_outer_V_sum += \
            np.outer(np.conj(tmp_logder_V.flatten()), tmp_logder_V.flatten()) 
        tmp_logder_V = np.conj(tmp_logder_V)
        logder_V_sum += tmp_logder_V 
        HO_V_sum += tmp_logder_V * tmp_energy 
        
        flat_logder_Q_sum += tmp_logder_Q.flatten() 
        logder_outer_Q_sum += \
            np.outer(np.conj(tmp_logder_Q.flatten()), tmp_logder_Q.flatten()) 
        tmp_logder_Q = np.conj(tmp_logder_Q)
        logder_Q_sum += tmp_logder_Q 
        HO_Q_sum += tmp_logder_Q * tmp_energy 
        
        flat_logder_K_sum += tmp_logder_K.flatten() 
        logder_outer_K_sum += \
            np.outer(np.conj(tmp_logder_K.flatten()), tmp_logder_K.flatten()) 
        tmp_logder_K = np.conj(tmp_logder_K)
        logder_K_sum += tmp_logder_K 
        HO_K_sum += tmp_logder_K * tmp_energy 
        
    energy_sum /= Nsample
    
    logder_W_sum /= Nsample
    HO_W_sum /= Nsample
    flat_logder_W_sum /= Nsample
    logder_outer_W_sum /= Nsample
    logder_outer_W_sum -= np.outer(np.conj(flat_logder_W_sum), flat_logder_W_sum)
    gradient_W = 2 * (HO_W_sum - logder_W_sum * energy_sum)
    grad_para_W = gradient_W.flatten()
    logder_outer_W_sum += np.eye(N**2) * L2_1
    deriv_W = np.linalg.solve(logder_outer_W_sum, grad_para_W)
    deriv_W = deriv_W.reshape((N, N))
    
    logder_V_sum /= Nsample
    HO_V_sum /= Nsample
    flat_logder_V_sum /= Nsample
    logder_outer_V_sum /= Nsample
    logder_outer_V_sum -= np.outer(np.conj(flat_logder_V_sum), flat_logder_V_sum)
    gradient_V = 2 * (HO_V_sum - logder_V_sum * energy_sum)
    grad_para_V = gradient_V.flatten()
    logder_outer_V_sum += np.eye(L**2) * L2_2
    deriv_V = np.linalg.solve(logder_outer_V_sum, grad_para_V)
    deriv_V = deriv_V.reshape((L, L))
    
    logder_Q_sum /= Nsample
    HO_Q_sum /= Nsample
    flat_logder_Q_sum /= Nsample
    logder_outer_Q_sum /= Nsample
    logder_outer_Q_sum -= np.outer(np.conj(flat_logder_Q_sum), flat_logder_Q_sum)
    gradient_Q = 2 * (HO_Q_sum - logder_Q_sum * energy_sum)
    grad_para_Q = gradient_Q.flatten()
    logder_outer_Q_sum += np.eye(L**2) * L2_2
    deriv_Q = np.linalg.solve(logder_outer_Q_sum, grad_para_Q)
    deriv_Q = deriv_Q.reshape((L, L))
    
    logder_K_sum /= Nsample
    HO_K_sum /= Nsample
    flat_logder_K_sum /= Nsample
    logder_outer_K_sum /= Nsample
    logder_outer_K_sum -= np.outer(np.conj(flat_logder_K_sum), flat_logder_K_sum)
    gradient_K = 2 * (HO_K_sum - logder_K_sum * energy_sum)
    grad_para_K = gradient_K.flatten()
    logder_outer_K_sum += np.eye(L**2) * L2_2
    deriv_K = np.linalg.solve(logder_outer_K_sum, grad_para_K)
    deriv_K = deriv_K.reshape((L, L))
    return energy_sum, deriv_Q, deriv_K, deriv_V, deriv_W

def get_fname(lam1,lam2,MARSHALL_SIGN, L2_1, L2_2, N, L, Nmc, num):
    return '_N=%i_L=%i_SIGN=%i_Nmc=%i'%(N,L,MARSHALL_SIGN,Nmc) + '_lam=(' \
        +"{:.0e}".format(lam1) + ', ' + "{:.0e}".format(lam2) +')' + \
         'L2=('+"{:.0e}".format(L2_1) + ', ' + "{:.0e}".format(L2_2) +')_%i'%num
 
def optimize( lam1, lam2, MARSHALL_SIGN, l21, l22, N,E0, L, Nop, Nmc, num):
    Q = np.random.rand(L, L) + 1j * np.random.rand(L, L)
    K = np.random.rand(L, L) + 1j * np.random.rand(L, L)
    W = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    V = np.random.rand(L, L) + 1j * np.random.rand(L, L)

    for i in range(Nop):
        E, gradQ, gradK, gradV, gradW = get_E_QKVW_MC_SR(Nmc, Q,K,V,W,MARSHALL_SIGN, L2_1, L2_2)

        W = W - lam1 * gradW
        V = V - lam2 * gradV
        Q = Q - lam2 * gradQ
        K = K - lam2 * gradK

        print('i = ', i, '\t E= ',E)
        
        if math.isnan(E.real):
            print('\n ERROR: ENERGY DIVERGES ')
            print('\n gradQ = \n ', gradQ)
            print('\n gradK = \n ', gradK)
            print('\n gradV = \n ', gradV)
            print('\n gradW = \n ', gradW)
            break
            return

        EE.append(E)
        
    pl.figure()
    pl.plot(np.real(EE), label='Nmc=%i'%Nmc)

    if MARSHALL_SIGN:
        pl.title('Monte Carlo SR \n psi(Q,K,V,W) \n \
                  Rotated Marshall sign (-) \n Nmc=%i \n  lam1=%.3f \n lam2=%.3f \n \
                      '%(Nmc, lam1,lam2) +'L2=('+"{:.0e}".format(L2_1) + ', ' + "{:.0e}".format(L2_2) +')' )
    else:
        pl.title('Monte Carlo SR \n psi(Q,K,V,W) \n \
                  Unrotated Marshall sign (+) \n Nmc=%i \n  lam1=%.3f \n lam2=%.3f \n \
                      '%(Nmc, lam1,lam2) +'L2=('+"{:.0e}".format(L2_1) + ', ' + "{:.0e}".format(L2_2) +')' )

    pl.ylabel('Energy')
    pl.xlabel('Iteration')
    pl.hlines(E0, 0, len(EE), color='r', linestyles='--', label='E=%.4f (N=%i)'%(E0,N))
    pl.legend()

    fname = get_fname(lam1,lam2,MARSHALL_SIGN,l21,l22,N,L,Nmc,num)

    fig_name = 'data/_FIG_001b.1' + fname + '.png'
    pl.savefig(fig_name)

    fname = 'data/001b.1' + fname + '.txt'
    np.savetxt(fname, EE)
    
lam1 = float(sys.argv[1])
lam2 = float(sys.argv[2])
N = int(sys.argv[3])
num = int(sys.argv[4])

EE = []
MARSHALL_SIGN = 0

L2_1 = 1e-3
L2_2 = 1e-2

Nmc = 500
nop = 1000
L = 2
Nc = N // L

E0 = -5.387389791340218 # energy for N=12

optimize(lam1, lam2, MARSHALL_SIGN, L2_1, L2_2, N,E0, L, nop, Nmc, num)

t=time.time()
print('\nRuntime:',(t-t0))

# test
