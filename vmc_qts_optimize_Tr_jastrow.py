import time
import math
import numpy as np
from numpy import exp, sqrt
from matplotlib import pyplot as pl

t0 = time.time()




def get_coeff(state, W):
    N = len(W[0,:])
    return np.exp(state.T @ W @ state)

def IBITS(n,i):
    return ((n >> i) & 1)

def _int_to_state2(integer, N):
    state = np.ones(N)
    for bit in range(N):
        if IBITS(integer,bit) == 0:
            state[bit] = 0
    return state


def _state_to_int(state, N):
    # x = (state+np.ones(N)*0.5)
    x=state
    x = x.astype('int')
    return int(''.join(map(str,x[::-1])),base=2 )

def rotate(state, n):
    return np.concatenate((state[n:], state[:n]))



    
def get_Tr_states(state):
    N = len(state)
    s = state+ np.ones(N)*0.5
    si = _state_to_int(s, N)
    si_dupes = []
    for r in range(N):
        sr = rotate(s, r)
        sri = _state_to_int(sr, N)
        if sri not in si_dupes:
        # if 1:
            si_dupes.append(sri)
    return si_dupes

def get_rep_state(state):
    N = len(state)
    s = state+ np.ones(N)*0.5 # state as 010101....
    si = _state_to_int(s, N)
    si_dupes = []
    m = 0 # multiplicity of state (how many times does state appear after N translations)
    for r in range(N):
        sr = rotate(s, r)
        sri = _state_to_int(sr, N)
        if sri not in si_dupes:
            si_dupes.append(sri)
        if sri == si:
            m += 1
    si = min(si_dupes)
    rep_state = _int_to_state2(si, N) - np.ones(N)*0.5
    # rep_state = _int_to_state2(si, N) 
    return rep_state, si_dupes, m


def spin_flip(state, si_dupes):
    N = len(state)
    found = 0

    while not found:
        x = np.random.randint(low=0,high=N)
        y=x
        while(state[y]*state[x] > 0):
            y = np.random.randint(low=0,high=N)
        new_state = state.copy()
        new_state[x] *= -1
        new_state[y] *= -1
    
        new_state, nsi_dupes, m = get_rep_state(new_state)
        nsi = min(nsi_dupes)
        for n in range(N):
            if state[n] != new_state[n]:
                found = 1
                break

    return new_state, nsi, nsi_dupes

N = 8
W = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))


import scipy as sc
from scipy.special import binom, comb

num_states = int(binom(N, N/2))

print('\nSystem size: N=',N)
print('Number of spin configurations: ', num_states )




state = np.ones(N)
state[:N//2] = -1
state *= 0.5
state = state[np.random.permutation(N)]

si_dupes = get_Tr_states(state)

rep_states = []
for i in range(1000):
    # state, si,  si_dupes = spin_flip(state, si_dupes)
    state = state[np.random.permutation(N)]
    # print(si, '\t', state+np.ones(N)*1/2)
    rep_state, si_dupes, m = get_rep_state(state)
    si = min(si_dupes)
    if si not in rep_states:
        rep_states.append(si)
        # print(si, '\t', state+np.ones(N)*1/2)
        
print('Number of representative states found: ', len(rep_states))

print('\n##############\n')




def get_eL(state, coeff, W):
    N = len(state)
    E1 = 0
    E2 = 0
    
    rep_state, si_dupes, m = get_rep_state(state)
    
    norm1 = np.sqrt(len(si_dupes)) # N1 / X1
    
    for i in range(N):
        E1 += state[i]*state[(i+1)%N]

        if (state[i]*state[(i+1)%N] < 0):
            state_new = state.copy()
            state_new[i] *= -1
            state_new[(i+1)%N] *= -1
            
            ns_dupes = get_Tr_states(state_new)
            si_new = min(ns_dupes)
            state_new = _int_to_state2(si_new, N) - np.ones(N)*0.5
            
            norm2 = np.sqrt(len(ns_dupes)) # N2 / X2
            
            
            coeff_new = get_coeff(state_new, W)  
            
            E2 += coeff_new / coeff


    return E1 + 0.5 * E2

def get_ALL_states(N):
    int_basis = []
    hbasis1D = []
    def bit_count(self):
        return bin(self).count("1")

    for i in range(2**N-1):
        # print(bit_count(i))
        # print(i)
        if bit_count(i) == int(N/2):
            # hbasis2D.append(i)
            int_basis.append(i)
            state = _int_to_state2(i,N)
            state = state - np.ones(N)*1/2
            hbasis1D.append(state)
    return hbasis1D, int_basis

states_list, int_list = get_ALL_states(N)


t = time.time()
print('\nTime to get states list: \t', (t-t0))
print(len(states_list))

print(rep_states)

rep_states_list = []

for si in rep_states:
    rep_state = _int_to_state2(si, N) - np.ones(N)*1/2
    rep_states_list.append(rep_state)
    
    
    


def get_exact_E(states_list, W):
    N = len(states_list[0])
    dtype=np.cfloat
    
    energy_sum = 0
    logder_W_sum = np.zeros((N,N), dtype= dtype)
    HO_W_sum = np.zeros((N,N), dtype= dtype)
    flat_logder_W_sum = np.zeros(N**2, dtype= dtype)
    logder_outer_W_sum = np.zeros((N**2, N**2), dtype= dtype)
    
    denom = 0
    for state in states_list:
        coeff = get_coeff(state, W)
        
        
        denom += np.abs(coeff)**2
        
        tmp_logder_W = np.outer(state,state)
        tmp_energy = get_eL(state, coeff, W) 
        
        energy_sum += np.conj( tmp_energy )* np.abs(coeff)**2
        
        flat_logder_W_sum += tmp_logder_W.flatten() 
        logder_outer_W_sum += \
            np.outer(np.conj(tmp_logder_W.flatten()), tmp_logder_W.flatten()) * np.abs(coeff)**2
            
        tmp_logder_W = np.conj(tmp_logder_W) 
        
        logder_W_sum += tmp_logder_W * np.abs(coeff)**2
        HO_W_sum += tmp_logder_W * tmp_energy * np.abs(coeff)**2
    
    energy_sum /= denom
    
    logder_W_sum /= denom
    HO_W_sum /= denom
    flat_logder_W_sum /= denom
    logder_outer_W_sum /= denom
    logder_outer_W_sum -= np.outer(np.conj(flat_logder_W_sum), flat_logder_W_sum)
    gradient_W = 2 * (HO_W_sum - logder_W_sum * energy_sum)
    grad_para_W = gradient_W.flatten()
    logder_outer_W_sum += np.eye(N**2) * 1e-2
    deriv_W = np.linalg.solve(logder_outer_W_sum, grad_para_W)
    deriv_W = deriv_W.reshape((N, N))
    
    return energy_sum, deriv_W
    # return energy_sum, gradient_W

if 0:
    # rep_states_list.copy()
    # energy_sum, deriv_W = get_exact_E(states_list, W)
    # print(rep_states_list)
    for s in rep_states_list:
        # print( s + np.ones(N)*1/2)
        rep_state, si_dupes, m = get_rep_state(s)
        si = min(si_dupes)
        print( rep_state + np.ones(N)*1/2, '\t', si, '\t', m)

    # new_state, nsi,  nsi_dupes = spin_flip(rep_state, si_dupes )
    

if 1:  
    
    pl.figure()
        
    for j in range(5):
        # N = 
        # L=2
        
    
        W = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))
    
        if N == 12:
            E0 =  -5.387389791340218 # right answer for N=12
        elif N == 8: 
            E0 = -3.6510934089371783 # right answer for N=8
        else:
            E0 = -0.4438 * N
            
    
        lam1 = 0.1
        
        EE = []
        for i in range(100):     
            E, gradW  = get_exact_E(rep_states_list, W)
            W = W - lam1 * gradW
    
            # print('i = ', i, '\t E= ',E)
            
            if math.isnan(E.real):
                print('\n ERROR: ENERGY DIVERGES ')
                print('\n gradW = \n ', gradW)
                break
            if i%150 == 0:
                W1 = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))
                W += W1/10
        
            EE.append(E)
            
        print('j = ', j, '\t E= ',E)
        pl.plot(np.real(EE), label='<E> run %i'%j)
     
    pl.title('Exact Solution - Tr Symm Reduced Basis')
    # pl.title('Exact Solution - Full Sz=0 Basis')
    pl.ylabel('Energy')
    pl.xlabel('Iteration')
    pl.hlines(E0, 0, len(EE), color='r', linestyles='--', label='E=%.4f (N=%i)'%(E0,N))
    pl.legend()
    
    
    
    
    
    
    
def metro(Nsample, W):
    N = len(W[0,:])
    # L = len(Q[0,:])
    
    dtype=np.cfloat
    
    logder_W_sum = np.zeros((N,N), dtype= dtype)
    HO_W_sum = np.zeros((N,N), dtype= dtype)
    flat_logder_W_sum = np.zeros(N**2, dtype= dtype)
    logder_outer_W_sum = np.zeros((N**2, N**2), dtype= dtype)
    

    energy_sum = 0
    
    state = np.ones(N)
    state[:N//2] = -1
    state *= 0.5
    state = state[np.random.permutation(N)]
    
    # for Tr symmetry
    si_dupes = get_Tr_states(state)
    si = min(si_dupes)
    state = _int_to_state2(si, N) - np.ones(N)*0.5
    # symm = 0
    
    state, si_dupes, m = get_rep_state(state)
    
    # Mn = np.sqrt(len(si_dupes))
    
    EE = []
    nacc = 0
    Nskip = 1
    for i in range(Nsample):
        for j in range(Nskip):

            coeff =  get_coeff(state, W) # gets <psi0|n >
            new_state, nsi, nsi_dupes = spin_flip(state, si_dupes) # gets spin flip into new representative
            # m = N // len(nsi_dupes)
            coeff_new = get_coeff(new_state, W)
            p = np.abs(coeff_new / coeff)**2
            # p /= m
            if (np.random.random() < min(1.0, p)):
                state = new_state.copy()
                coeff = coeff_new
                si_dupes = nsi_dupes.copy()
                nacc += 1

                
        tmp_energy = get_eL(state, coeff, W)
        tmp_logder_W = np.outer(state,state)
        energy_sum += tmp_energy 
        
        flat_logder_W_sum += tmp_logder_W.flatten() 
        logder_outer_W_sum += \
            np.outer(np.conj(tmp_logder_W.flatten()), tmp_logder_W.flatten()) 
        tmp_logder_W = np.conj(tmp_logder_W)
        logder_W_sum += tmp_logder_W 
        HO_W_sum += tmp_logder_W * tmp_energy 
        
        
        EE.append(energy_sum/(i+1))
        
    energy_sum /= Nsample
    
    logder_W_sum /= Nsample
    HO_W_sum /= Nsample
    flat_logder_W_sum /= Nsample
    logder_outer_W_sum /= Nsample
    logder_outer_W_sum -= np.outer(np.conj(flat_logder_W_sum), flat_logder_W_sum)
    gradient_W = 2 * (HO_W_sum - logder_W_sum * energy_sum)
    grad_para_W = gradient_W.flatten()
    logder_outer_W_sum += np.eye(N**2) * 1e-3
    deriv_W = np.linalg.solve(logder_outer_W_sum, grad_para_W)
    deriv_W = deriv_W.reshape((N, N))
   
    
    if 0:
        pl.figure()
        pl.plot(EE)
        pl.show()
    # print('acceptance ratio = ', nacc/Nsample/Nskip)
    racc = nacc/Nsample/Nskip
    return energy_sum, deriv_W, racc

if 0:  
    state = np.ones(N)
    state[:N//2] = -1
    state *= 0.5
    state = state[np.random.permutation(N)]
    
    state, si_dupes, m = get_rep_state(state)
    m_old = N // len(si_dupes)
    
    prob_dist = []
    for i in range(10000):
        # new_state, nsi, nsi_dupes = spin_flip(state, si_dupes) # gets spin flip into new representative
        
        new_state = state[np.random.permutation(N)]
        new_state, si_dupes, m = get_rep_state(new_state)
        nsi = min(si_dupes)
        
        # m_new = N // len(nsi_dupes)
        # coeff_new = get_coeff(new_state, W)
        p = 1
        # p /= m_new
        # p *= m_old/ m_new
        if (np.random.random() < min(1.0, p)):
            state = new_state.copy()
            prob_dist.append(nsi)
            # m_old = m_new
    
  
    
    
    rep_states.sort()
    for i in rep_states:
        print(i , '\t', prob_dist.count(i))
        
        
    pl.figure()
    pl.hist(prob_dist, bins=100, rwidth=0.9)
    pl.grid()
    pl.show()

if 0:  
    N = 8
    L=2
    

    W = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))

    
    if N == 12:
        E0 =  -5.387389791340218 # right answer for N=12
    elif N == 8: 
        E0 = -3.6510934089371783 # right answer for N=8
    else:
        E0 = -0.4438 * N
        
    Nmc = 400
    lam1 = 0.02
    # lam2 = 0.005
    
    EE = []
    for i in range(100):     
        E, gradW, racc = metro(Nmc, W)
        W = W -lam1 * gradW
        print('i = ', i,'\t E= ',E)
        
        if math.isnan(E.real):
            print('\n ERROR: ENERGY DIVERGES ')
            print('\n gradW = \n ', gradW)
            break
        if i%50 == 0:
            W1 = np.random.uniform(low=-1, high=1, size=(N,N)) + 1j*np.random.uniform(low=-1, high=1, size=(N,N))
            W += W1/5

        EE.append(E)
        
    pl.figure()
    pl.plot(np.real(EE), label='Nmc=%i'%Nmc)
    
    pl.ylabel('Energy')
    pl.xlabel('Iteration')
    pl.title('Monte Carlo Reduced Basis')
    pl.hlines(E0, 0, len(EE), color='r', linestyles='--', label='E=%.4f (N=%i)'%(E0,N))
    pl.legend()

def get_fname(lam1,lam2, L2_1, L2_2, N, L, Nmc, num):
    return 'N=%i_L=%i'%(N,L) + '_lam=(' \
        +"{:.0e}".format(lam1) + ', ' + "{:.0e}".format(lam2) +')' + \
         'L2=('+"{:.0e}".format(L2_1) + ', ' + "{:.0e}".format(L2_2) +')_%i'%num+ '_Nmc=%i_'%Nmc + '%i'%num


    
# lam1 = float(sys.argv[1])
# lam2 = float(sys.argv[2])
# N = int(sys.argv[3])
# num = int(sys.argv[4])


EE = []
lam1 = 0.01
lam2 = 0.01

L2_1 = 1e-3
L2_2 = 1e-3

Nmc = 200
nop = 50

N = 8
L = 2

MARSHALL_SIGN = 0 
num = 0





t=time.time()
print('\nRuntime:',(t-t0))