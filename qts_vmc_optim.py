#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:39:59 2024

@author: ashilcusky
"""

import sys
import time
import math
import numpy as np
from numpy import exp, sqrt
from matplotlib import pyplot as pl

t0 = time.time()

# lam1 = float(sys.argv[1]) 
# lam2 = float(sys.argv[2])
# N = int(sys.argv[3])
# Nmc = int(sys.argv[4])
# num = int(sys.argv[5])
N = 20
# 

def calc_dQdK(aI, zII, zIJ, xI, xlist, Q, K):
    L = len(Q[0,:])
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
    N = len(W[0,:])
    L = len(Q[0,:])
    Nc = N // L
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
    return alist, z, coeff, vtilde

def get_logderQK_josh(state, Q,K,V,W, aI, z):
    N = len(W[:,0])
    L = len(Q[:,0])
    Nc = N//L
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
    
    return Qder, Kder

def get_logderV(state, aI, V, W):
    # dtype=complex
    N = len(W[:,0])
    L = len(V[:,0])
    Nc = N//L
    xlist = state.reshape(Nc, L)
    
    aJv = (aI[:, np.newaxis] * np.matmul(xlist, V.T)).reshape(N)

    reorder = list(range(0, N, 2)) + list(range(1, N, 2))

    aIxI = (aI[:, np.newaxis] * xlist).reshape(N)[reorder]

    WaJv = (W @ aJv)[reorder]
    aJvW = (aJv @ W)[reorder]

    WaJv_sets = [WaJv[:N//2], WaJv[N//2:]] # evens, odds
    aJvW_sets = [aJvW[:N//2], aJvW[N//2:]] # evens, odds
    aIxI_sets = [aIxI[:N//2], aIxI[N//2:]] # evens, odds

    logder_V = np.zeros((L, L), dtype=complex)
    for i in range(L):
        for j in range(L):
            logder_V[i, j] = aIxI_sets[j].dot(WaJv_sets[i]) \
                           + aJvW_sets[i].dot(aIxI_sets[j])
    return logder_V

def get_eL(state, coeff, Q,K,V,W, MARSHALL_SIGN):
    N = len(state)
    E1 = 0
    E2 = 0
    for i in range(N):
        E1 += state[i]*state[(i+1)%N]
        if (state[i]*state[(i+1)%N] < 0):
            # state_new = state.copy()
            state_new = state.copy()
            state_new[i] *= -1
            state_new[(i+1)%N] *= -1
            
            logpsi = calc_aI(state_new, Q, K, V, W)[2]
            coeff_new = np.exp(logpsi)
            
            E2 += coeff_new/coeff
    if MARSHALL_SIGN:
        return E1 - 0.5*E2
    else:
        return E1 + 0.5*E2
    

def get_E_QKVW_MC_SR(Nsample, Q,K,V,W, MARSHALL_SIGN, L2_1, L2_2):
    N = len(W[0,:])
    L = len(Q[0,:])
    energy_sum = 0.0

    dtype=complex
    
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
    
    # print(logder_outer_W_sum.dtype)
    
    EE = []
    
    state = np.ones(N)
    state[:N//2] = -1
    state *= 0.5
    state = state[np.random.permutation(N)]
    for i in range(Nsample):
        for j in range(3):
            x = np.random.randint(low=0,high=N)
            y=x
            while(state[y]*state[x] > 0):
                y = np.random.randint(low=0,high=N)
            new_state = state.copy()
            new_state[x] *= -1
            new_state[y] *= -1
            
            aI, z, logpsi, vtilde = calc_aI(state, Q, K, V, W)
            coeff = np.exp(logpsi)
            aI_new, z_new, logpsi_new, vtilde_new = calc_aI(new_state, Q, K, V, W)
            coeff_new = np.exp(logpsi_new)
        
            
            
            if (np.random.random() < min(1.0, np.abs(coeff_new/coeff)**2)):
                state = new_state.copy()
                coeff = coeff_new
                aI = aI_new.copy()
                z = z_new.copy()
                vtilde = vtilde_new
            
        tmp_energy = get_eL(state,coeff, Q,K,V,W, MARSHALL_SIGN)
        
        tmp_logder_Q, tmp_logder_K = get_logderQK_josh(state, Q,K,V,W, aI, z)
        tmp_logder_W = np.outer(vtilde, vtilde)
        tmp_logder_V = get_logderV(state, aI, V, W)
        
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
        
        EE.append(energy_sum/(i+1))
        
        
    energy_sum /= Nsample
    
    
    logder_W_sum /= Nsample
    HO_W_sum /= Nsample
    flat_logder_W_sum /= Nsample
    logder_outer_W_sum /= Nsample
    logder_outer_W_sum -= np.outer(np.conj(flat_logder_W_sum), flat_logder_W_sum)
    gradient_W = 2*(HO_W_sum - logder_W_sum * energy_sum)
    grad_para_W = gradient_W.flatten()
    logder_outer_W_sum += np.eye(N**2) * L2_1
    deriv_W = np.linalg.solve(logder_outer_W_sum, grad_para_W)
    # deriv_W = np.linalg.lstsq(logder_outer_W_sum, grad_para_W)[0]
    deriv_W = deriv_W.reshape((N,N))
    
    
    
    logder_V_sum /= Nsample
    HO_V_sum /= Nsample
    flat_logder_V_sum /= Nsample
    logder_outer_V_sum /= Nsample
    logder_outer_V_sum -= np.outer(np.conj(flat_logder_V_sum), flat_logder_V_sum)
    gradient_V = 2*(HO_V_sum - logder_V_sum * energy_sum)
    grad_para_V = gradient_V.flatten()
    logder_outer_V_sum += np.eye(L**2) * L2_2
    deriv_V = np.linalg.solve(logder_outer_V_sum, grad_para_V)
    # deriv_V = np.linalg.lstsq(logder_outer_V_sum, grad_para_V)[0]
    deriv_V = deriv_V.reshape((L,L))
    
    
    
    logder_Q_sum /= Nsample
    HO_Q_sum /= Nsample
    flat_logder_Q_sum /= Nsample
    logder_outer_Q_sum /= Nsample
    logder_outer_Q_sum -= np.outer(np.conj(flat_logder_Q_sum), flat_logder_Q_sum)
    gradient_Q = 2*(HO_Q_sum - logder_Q_sum * energy_sum)
    grad_para_Q = gradient_Q.flatten()
    logder_outer_Q_sum += np.eye(L**2) * L2_2
    deriv_Q = np.linalg.solve(logder_outer_Q_sum, grad_para_Q)
    # deriv_Q = np.linalg.lstsq(logder_outer_Q_sum, grad_para_Q)[0]
    deriv_Q = deriv_Q.reshape((L,L))
    
    
    
    logder_K_sum /= Nsample
    HO_K_sum /= Nsample
    flat_logder_K_sum /= Nsample
    logder_outer_K_sum /= Nsample
    logder_outer_K_sum -= np.outer(np.conj(flat_logder_K_sum), flat_logder_K_sum)
    gradient_K = 2*(HO_K_sum - logder_K_sum * energy_sum)
    grad_para_K = gradient_K.flatten()
    logder_outer_K_sum += np.eye(L**2) * L2_2
    deriv_K = np.linalg.solve(logder_outer_K_sum, grad_para_K)
    # deriv_K = np.linalg.lstsq(logder_outer_K_sum, grad_para_K)[0]
    deriv_K = deriv_K.reshape((L,L))
    
    # pl.figure()
    # pl.plot(EE)
    # pl.show()
    
    return energy_sum, deriv_Q, deriv_K, deriv_V, deriv_W



def get_fname(lam1,lam2,MARSHALL_SIGN, L2_1, L2_2, N, L, Nmc, num):
    return 'N=%i_L=%i_SIGN=%i_Nmc=%i'%(N,L,MARSHALL_SIGN,Nmc) + '_lam=(' \
        +"{:.0e}".format(lam1) + ', ' + "{:.0e}".format(lam2) +')' + \
         'L2=('+"{:.0e}".format(L2_1) + ', ' + "{:.0e}".format(L2_2) +')_%i'%num

if N == 12:
    E0 =  -5.387389791340218 # right answer for N=12
elif N == 8: 
    E0 = -3.6510934089371783 # right answer for N=8
else:
    E0 = -0.4438 * N
    
    
L = 2

#Q = np.random.uniform(low=-1, high=1, size=(L,L)) + 1j*np.random.uniform(low=-1, high=1, size=(L,L))
#K = np.random.uniform(low=-1, high=1, size=(L,L)) + 1j*np.random.uniform(low=-1, high=1, size=(L,L))
V = np.random.uniform(low=-1, high=1, size=(L,L)) + 1j*np.random.uniform(low=-1, high=1, size=(L,L))
W = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))

Q = np.random.uniform(low=-1, high=1, size=(L,L)) 
K = np.random.uniform(low=-1, high=1, size=(L,L)) 

num = 0

if 1: 
    ''' Testing for SR with Monte Carlo using Q,K,V,W '''
    print()
    lam1 = 0.05
    lam2 = 0.005
    # lam = 0.5
    # lam2 = 0.5
    EE=[]
    MARSHALL_SIGN = 0
    Nmc = 2000
    
    L2_1 = 1e-3
    L2_2 = 1e-3
    for i in range(100):
        E, gradQ, gradK, gradV, gradW = get_E_QKVW_MC_SR(Nmc, Q,K,V,W,MARSHALL_SIGN, L2_1, L2_2)
        W = W -lam1 * gradW
        V = V -lam2 * gradV
        Q = Q -lam2 * gradQ
        K = K -lam2 * gradK
        # print('i = ', i, '\t E= ',E, '\t gradQ[0,0] =', gradQ[0,0])
        print('i = ', i, '\t E= ',E)
        if math.isnan(E.real):
            print('\n ERROR: ENERGY DIVERGES ')
            print('\n gradQ = \n ', gradQ)
            print('\n gradK = \n ', gradK)
            print('\n gradV = \n ', gradV)
            print('\n gradW = \n ', gradW)
            break
        if i %150 == 0:
            #Q1 = np.random.uniform(low=-1, high=1, size=(L,L)) + 1j*np.random.uniform(low=-1, high=1, size=(L,L))
            #K1 = np.random.uniform(low=-1, high=1, size=(L,L)) + 1j*np.random.uniform(low=-1, high=1, size=(L,L))
            V1 = np.random.uniform(low=-1, high=1, size=(L,L)) + 1j*np.random.uniform(low=-1, high=1, size=(L,L))
            W1 = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))
            Q1 = np.random.uniform(low=-1, high=1, size=(L,L)) 
            K1 = np.random.uniform(low=-1, high=1, size=(L,L))
            Q += Q1/10
            K += K1/10
            W += W1/10
            V += V1/10
        EE.append(E)

    t1 = time.time()
    t = t1-t0
    t = t/60/60
    
    pl.figure()
    pl.plot(np.real(EE), label='Nmc=%i'%Nmc)
    
    pl.ylabel('Energy')
    pl.xlabel('Iteration')
    pl.hlines(E0, 0, len(EE), color='r', linestyles='--', label='E=%.4f (N=%i)'%(E0,N))
    pl.legend()
    fname = get_fname(lam1,lam2,MARSHALL_SIGN,L2_1,L2_2,N,L,Nmc,num)
    pl.title('lam1=%.3f, lam2=%.3f \nRuntime: %.2f hrs'%(lam1,lam2, t))
    fig_name = 'data/' + fname + '.png'
    pl.savefig(fig_name)
    fname = 'data/' + fname + '.txt'
    np.savetxt(fname, EE)
    
    
t = time.time()

print('\nRuntime: ', (t-t0))
# import jax
# import jax.numpy as jnp

# N = 20
# L = 2
# Nc = N // L

# W = np.random.rand(N, N) + 1j * np.random.rand(N, N)
# Q = np.random.rand(L, L) + 1j * np.random.rand(L, L)
# K = np.random.rand(L, L) + 1j * np.random.rand(L, L)
# V = np.random.rand(L, L) + 1j * np.random.rand(L, L)

# state = np.ones(N)
# state[:N//2] = -1
# state *= 0.5
# state = state[np.random.permutation(N)]


# def get_coeff_2(state,Q,K,V,W):
#     N = len(W[:,0])
#     L = len(Q[:,0])
#     Nc = N // L
#     xlist = np.reshape(state,(Nc,L))
    
#     Qx = jnp.matmul(xlist, Q.T)
#     Kx = jnp.matmul(xlist, K.T)
#     # z = jnp.matmul(jnp.conj(Qx), Kx.T)/np.sqrt(L)
#     z = jnp.matmul(Qx, Kx.T)/np.sqrt(L)
    
#     Vx = jnp.matmul(xlist, V.T)
    
#     sftmx = jax.nn.softmax(-z)
#     a = jnp.diag(sftmx)
#     aa = jnp.repeat(a, L)
    
#     vtilde = jnp.reshape(Vx, (N))
#     vtilde = aa * vtilde
#     # return jnp.exp(vtilde.T @ W @ vtilde)
#     return vtilde.T @ W @ vtilde

# aI, z, coeff, vtilde = calc_aI(state, Q, K, V, W)
# derQ, derK = get_logderQK_josh(state, Q,K,V,W, aI, z)

# tmp_der_Q, tmp_der_K, tmp_der_V, tmp_der_W \
#         = jax.grad(get_coeff_2, holomorphic=1, argnums=(1,2,3,4))(state,Q,K,V,W)
        
        
# print(derQ)
# print(tmp_der_Q)

# derV = get_logderV(state,Q,K,V,W,aI)

# print(derV)
# print(tmp_der_V)
# t1=time.time()
# c = get_coeff_2(state,Q,K,V,W)
# t2 = time.time()

# tjax = t2-t1

# t1=time.time()
# aI, z, c1, vtilde = calc_aI(state, Q, K, V, W)
# t2 = time.time()

# tjosh = t2-t1      
