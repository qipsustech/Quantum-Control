#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve the dynamics, these classes are usually called by Solver.py

@author: junningli
"""
import copy as cp
from scipy.integrate import solve_ivp
import qutip as qt
import numpy as np

 
class Sesolver(object):
        
    def solve(self, H, tlist, u0=None, dynamics='expm'):
        '''
        H: array of hamitonian qobj of dynamics
        tlist: time array of dynamics
        u0 here should be Qobj
        dynamics: method of dynamcs. expm or ode for this version
        '''
        self.H = cp.deepcopy(H)
        self.tlist = cp.deepcopy(tlist)

        if u0 != None:
            self.u0 = cp.deepcopy(u0)
        else:
            self.u0 = qt.qeye(self.H[0].shape[0])
        
        if dynamics == 'expm':
            self.sesolver_expm()
        elif dynamics == 'ode':
            self.sesolver_ode()
        else:
            raise ValueError('Please check your dynamical method !')
        
    '''
    ==========directly solve with expm=========
    '''
    def sesolver_expm(self):
        
        u = [self.u0]
        
        for i in range(1, len(self.tlist)):      
            ut = (-1j * self.H[i] * (self.tlist[i] - self.tlist[i-1])).expm()
            u.append(ut * u[i-1])  

        self.U = np.array(u, dtype=qt.Qobj)

    '''
    =========ode function from SciPy===============
    '''
    def sesolver_ode(self):
        '''
        Solve the schordinger's eq for U, with Scipy.ode
        one-step
        '''
        ushape = self.u0.shape
        u0reshape = np.reshape(self.u0.full(), np.prod(ushape))

        self.U= [self.u0]
        
        for i in range(1, len(self.tlist)):      
            self.H_ = self.H[i]
    
            r = solve_ivp(self.func, [self.tlist[i-1], self.tlist[i]], u0reshape) 
            
            u0reshape = cp.deepcopy(np.transpose(r.y)[-1])
            self.U.append(qt.Qobj(np.reshape(u0reshape, ushape)))
            
        self.U = np.array(self.U, dtype=qt.Qboj)
    
    def func(self, t, y):       
        y_ = np.reshape(y, self.H_.shape)
        f = -1j * np.dot(self.H_.full(), y_)       
        return np.reshape(f, y.shape)









    '''
    The following code is abandoned!
    '''
    '''
    =========manually solver with RK4=============
    '''
    def sesolver_rk4(self, t0=0.0, n=10):
        '''
        Solve the schordinger's eq for U, with RK4
        return: array of ut(len), len = nt
        '''                    
        #dim = H(0).shape[0]
        f = lambda t, u: -1j * self.H(t) * u
        h = self.inp.dt / n

        u= [self.u0]
        t = cp.deepcopy(t0)
        
        ut, t = self.Euler_step(f, t, self.u0, h)     #step 1 with Euler
        if n == 1: u.append(ut)        #record step 1
        
        while t < self.inp.tg-0.9 * self.inp.dt:
            
            ut, t = self.RK4_step(f, t, ut, h)
            
            if abs(t - self.tlist[len(u)-1]-self.inp.dt) < (0.5 * h):
                u.append(ut)
        
        self.u = np.array(u)
    
    def RK4_step(self, f, t, y0, h):
        dy1 = f(t    , y0            )
        dy2 = f(t+h  , y0 + h/2 * dy1)
        dy3 = f(t+h/2, y0 + h/2 * dy2)
        dy4 = f(t+h  , y0 + h   * dy3)
        dy = h * (dy1 + 2*dy2 + 2*dy3 + dy4) / 6
        return (y0 + dy), (t + h)
    
    def Euler_step(self, f, t, y0, h):
        y_ = y0 + h * f(t, y0)
        y = y0 + h * (f(t, y0) + f(t+h, y_))/2
        return y, (t + h)
        
    
    def solve_step(self, Hi, it, u0):
        '''
        ======choose the solver method here===============
        u0: Qobj
        '''
        self.it = cp.deepcopy(it)
        self.H[self.it] = cp.deepcopy(Hi)
        self.t0 = cp.deepcopy(self.tlist[it]) #it * self.inp.dt
        self.u0 = cp.deepcopy(u0.full()) #tranform to array

        
        if self.method == 'expm':
            self.sesolver_expm()
            
        elif self.method == 'rk4':
            self.sesolver_rk4()
            
        elif self.method == 'ode':
            self.sesolver_ode()
            
        else:
            raise ('Please type the correct method!')
            
    def solve_all(self, H):
        self.t0 = self.tlist[0]
        self.H = cp.deepcopy(H)
        u0 = cp.deepcopy(self.inp.u0)
        self.u_arr = np.array([u0 for i in range(self.inp.nt)])
  
        for i in range(self.inp.nt-1):
            self.solve_step(self.H[i], i, self.u_arr[i])    
            self.u_arr[i+1] = cp.deepcopy(self.u)
            


