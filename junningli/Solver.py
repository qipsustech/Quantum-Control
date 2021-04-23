#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a Physical solver for circuit QED

v0.9
Updated on 04/20/2021
@author: junningli
"""
'''
FUNCTIONS:
    
    define_qubit: define a qubit with frequency, and number of energy levels in SIM if applicable
    
    couple_qubits_x: couple a list of qubits with the XX coupling strength matrix g 
    
    unitary_fidelity: calculate the gate fidelity
    
OBJECTS:
    
    Solver_cqed: the solver for circuit QED
    
'''
import qutip as qt
import numpy as np
import copy as cp
import Dynamics 


def define_qubit(frequency, levels=2, eta=0.0):
    '''
    

    Parameters
    ----------
    frequency : qubit frequency
    levels : number of qubit levels to calculate. The default is 2.
    eta : inharmonicity of transmon. The default is 0.0.

    Returns
    -------
    Qobj of the qubit under fock space

    '''
    return 2 * np.pi * (frequency * qt.create(levels) * qt.destroy(levels) \
        - eta/2 * qt.create(levels) * qt.create(levels) * qt.destroy(levels) * qt.destroy(levels)) 
        
def couple_qubits_x(qubits, g, ctype='xx'):
    '''
    
    
    Parameters
    ----------
    qubits : a list or 1darray of qobjs
        Qubits to be coupled in order.
    g : float or matrix of float
        Single number represents all qubits have the same coupling, 
        the dimension of array should be the same as number of qubits number.
    ctype : 'xx' or 'zz', optional
        The default is 'xx'.

    Returns
    -------
    an Qobj of coupled qubits system in total fock space

    '''
    if type(g) == float:
        gs = np.zeros((len(qubits), len(qubits)))
        gs[:] = g * 2 * np.pi
    elif len(g[0]) == len(qubits) and len(g[1]) == len(qubits):
        gs = g * 2 * np.pi
    else:
        raise ValueError('g should be float or nd matrix!')
    
    qblevels = []
    for i in range(len(qubits)):
        qblevels.append(qubits[i].dims[0][0])
        
    bdag, b, hi = [], [], []
    H0 = 0.0
    for nj in range(len(qubits)): 
        opbdag, opb, oph = [], [], []
        for k in range(len(qubits)): 
            if k == nj:
                opbdag.append(qt.create(qblevels[k]))
                opb.append(qt.destroy(qblevels[k]))
                oph.append(qubits[k])
            else:
                opbdag.append(qt.qeye(qblevels[k]))
                opb.append(qt.qeye(qblevels[k]))
                oph.append(qt.qeye(qblevels[k]))
     
        bdag.append(qt.tensor(opbdag))
        b.append(qt.tensor(opb))
        hi.append(qt.tensor(oph))
        
    for i in range(len(qubits)):
        H0 += hi[i]
        for j in range(i):
            if ctype == 'xx':#XX
                H0 += gs[i][j] * (bdag[i] * b[j] + b[i] * bdag[j])
            elif ctype == 'zz':#ZZ
                H0 += gs[i][j] * bdag[i] * b[i] * bdag[j] * b[j]
                
    return H0   

def unitary_fidelity(u_calc, u_id):
    '''
    calculate the fidelity between two unitaries
    u_calc and u_target should be Qobjs
    '''
    
    ndim = u_calc.shape[0]
    M = u_id * u_calc.dag()
    MMdag = M * M.dag()
    #Mt = np.trace(np.matmul(u_id.full(), u_calc.dag().full()))
    F = 1 / (ndim * (ndim + 1)) * (MMdag.tr() + abs(M.tr()) ** 2)
    return F

class Solver_cqed(object):
    def __init__(self, H0=None, dynamics='expm'):
        '''
        Solver object for cqed system
        
        Parameters
        ----------
        H0 : Qobj, optional
            The hamitonian of qubit system without drive. You may alsodefine it
            with function import_H0
        dynamics : String, optional
            Method of dynamics, should be expm or ode. The default is 'expm'.

        '''
        if H0 != None: self.import_H0(H0)
        self.dynamics = dynamics
                
    def import_H0(self, H0):
        '''
        Import system Hamiltonian without drive term

        Parameters
        ----------
        H0 : Qobj
            Hamiltonian without drive terms

        '''
        self.H0 = H0
        self._operators()
        
        self.Hdiag, self.T = diag(self.H0)
        
        if (self.qblevels > 2).any():
            self.Hdiag, self.T, self.eigenE = rearrange_levels(self.Hdiag, self.T, self.qblevels)
        elif (self.qblevels < 2).any():
            raise ValueError('There is an qubit level less than 2 !')
            
    def _operators(self):
        self.qblevels = np.array(self.H0.dims[0])
        self.nqb = len(self.qblevels)
        self.bdag, self.b = [], []
        for nj in range(self.nqb): 
            opbdag, opb = [], []
            for k in range(self.nqb): 
                if k == nj:
                    opbdag.append(qt.create(self.qblevels[k]))
                    opb.append(qt.destroy(self.qblevels[k]))
                else:
                    opbdag.append(qt.qeye(self.qblevels[k]))
                    opb.append(qt.qeye(self.qblevels[k]))
         
            self.bdag.append(qt.tensor(opbdag))
            self.b.append(qt.tensor(opb))
            
    def generate_drive_term(self, tlist, wd, drive_qb=1):
        '''
        generate the drive term for transmon. Note that the driven H
        H = H0_int + pulse * Hd

        Parameters
        ----------
        tlist : 1darray
            array of the evolution time
        wd : float
            drive frequency with unit of 2pi * GHz
        drive_qb : integer, optional
            Number of the driven qubit. The default is 1.

        Returns
        -------
        1d array of Objs of Hd.

        '''
        dq = drive_qb - 1
        hd = []
        for i in range(len(tlist)):
            hd.append(self.eigen_transform(
                                np.exp(-1j * wd * tlist[i]) * self.bdag[dq] + 
                                np.exp( 1j * wd * tlist[i]) * self.b[dq]
                        )
                )
            
        hd = np.array(hd, dtype=qt.Qobj)
        
        H0, H1 = [], []
        for i in range(len(tlist)):
            h0_, h1_ = int_trans(self.Hdiag, hd[i], tlist[i])             
            H0.append(h0_)
            H1.append(h1_)
            
        self.H0_int = np.array(H0, dtype=qt.Qobj)
        self.Hd = np.array(H1, dtype=qt.Qobj)
        
        return self.Hd   
        
    def eigen_transform(self, operator):
        '''
        transform the operator from original representation to eigen-space

        Parameters
        ----------
        operator : Qobj
            Full size of the Hillbert space as the calculated qubit system 

        Returns
        -------
        Oobj

        '''
        return transT(operator, self.T)
        
    def dipoles_2qb(self):
        elem_num = np.array([0, 1, 
                             self.qblevels[0], self.qblevels[0]+1])
        self.dy = np.zeros((2,2))

        X1_eig_sub = O_subspace(transT(self.bdag[0] + self.b[0], self.T), elem_num)
        X2_eig_sub = O_subspace(transT(self.bdag[1] + self.b[1], self.T), elem_num)
        
        self.dy[0][0] = X1_eig_sub[0, 2]
        self.dy[0][1] = X1_eig_sub[1, 3]
        self.dy[1][0] = X2_eig_sub[0, 1]
        self.dy[1][1] = X2_eig_sub[2, 3]
        
        return self.dy
    
    def unitary_dynamics(self, pulse, tlist, wd, drive_qb=1, dynamics=None):
        '''
        User may call this function after created solver object
        
        pulse: the pulse envelope of drive
        tlist: time array of drive
        wd: drive frequency
        drive_qb: drive qubit, start from 1, instead of 0
        '''
        if len(pulse) != len(tlist): 
            raise ValueError('pulse array should have the same length with t array!')
            
        self.generate_drive_term(tlist, wd, drive_qb)

        #self.H = np.array([self.Hdiag for i in range(len(tlist))], dtype=qt.Qobj) 
        self.H = self.H0_int + pulse * self.Hd
        
        if dynamics == None: dynamics = self.dynamics
        desolver = Dynamics.Sesolver()
        desolver.solve(self.H, tlist, dynamics=dynamics)
        self.U = desolver.U   

'''=============================
Following are functions used in the solver objects
==============================='''        

def diag(H0):
    '''
    H is the H0 matrix, not a function
    return: diagonalised H0 and corresponding T, while
    Hdiag = Tdag * H * T
    '''
    eigE, eigS = np.linalg.eigh(H0.full())

    Hdiag = np.diag(eigE)    
    Hq, Tq = qt.Qobj(Hdiag), qt.Qobj(eigS)
    return Hq, Tq# , eigE

def transT(O, T):
    '''
    Basis transformation, T matrix is made with column eigenvectors
    '''
    if O.dims != T.dims: O.dims =T.dims
    return T.dag() * O * T

def O_subspace(O_, elem_num):
    '''O_ should be qobj'''
    dim = len(elem_num)
    Omat = O_.full()
    Ocs = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            Ocs[i][j] = Omat[elem_num[i], elem_num[j]]
            
    return qt.Qobj(Ocs)

def rearrange_levels(Hdiag, Tq, nlevels):
    '''
    eigE: eigenenergies from low to high
    Tq: qobj of the diagonalization matrix of H0, columns are eigenvectors
    nlevels: tow elements list of two qubit level numbers
    '''
    #n1, n2 = nlevels[0]-1, nlevels[1]-1
    eigE = Hdiag.diag()
    energies, vectors = [], []
    T = Tq.full()
    Tt = T.transpose()

    argmax_num = abs(T).argmax(0) #indexes of the max elem for each column
    #search and change if we have repeated argnumbers
    for i in range(len(argmax_num)):
        argvec = np.argsort(abs(Tt[i]))
        k = -1
        j = 0
        #if argvec[-1] != argmax_num[i]: 
        #    raise ValueError('Your code is not right!')
        while j < i:
            if argmax_num[i] == argmax_num[j]:
                if abs(k) > len(argvec): 
                    raise ValueError('found an error in rearranging levels!')         
                
                argmax_num[i] = cp.deepcopy(argvec[k])
                k -= 1
                j = 0
            else:
                j += 1
    
    for i in range(len(argmax_num)):
        for n in range(len(argmax_num)):
            if argmax_num[n] == i: 
                energies.append(eigE[n])
                
                if abs(Tt[n][i]) != Tt[n][i]:
                    vectors.append(-Tt[n])
                else:
                    vectors.append(Tt[n])
                
    if len(argmax_num) == np.prod(nlevels) and len(energies) == np.prod(nlevels):
        return qt.Qobj(np.diag(np.array(energies))), qt.Qobj(np.array(vectors).transpose()), np.array(energies)
    else:
        raise ValueError('Error in rearangement!')
        
def int_trans(H0, H1, t):
    '''
    change into the interaction picture
    H0+H1 is Hamiltonian in t
    return H_ = RIdag * H * R + i 
    '''
    R = (1j * H0 * t).expm()
    Rdag = R.dag()
    dRdag = -1j * H0 * Rdag
    
    H_0 =  R * H0 * Rdag - 1j * R * dRdag
    H_1 = R * H1 * Rdag
    
    return H_0, H_1        

