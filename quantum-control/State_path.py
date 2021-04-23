#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a class of Bloch sphere plotting, from the unitary operator array.
data_path should be changed to the figure saving directory 

@author: junning
"""
 
import qutip as qt
import numpy as np
import copy as cp

'''
some default settings for plot
'''
fig_params = {
                'xlabel': ['$\sigma_x$',''],
                'ylabel': ['$\sigma_y$',''],
                'view': [-150,30]
                }

def np2qobj(npmatrix):
    return qt.Qobj(npmatrix)

class plot_bloch_1qb(object):
    def __init__(self):
        self.dm0 = qt.ket2dm(qt.basis(2,0)) 
        self.dm1 = qt.ket2dm(qt.basis(2,1)) 

        self.bloch = qt.Bloch(view=fig_params['view'])
        #self.fig_parms = inp.fig_params_bloch
        
    def plot(self, u_arr, data_path, name='bloch'):
        '''
        u_arr: array of Qobjs of two-qubit-gates
        s0: Qobj of two qubit state
        plot the two qubit states trajectories in one sphere
        '''
        print('Start plotting bloch spheres.')
        self.u_arr = cp.deepcopy(u_arr)
        
        self.dm2bloch(self.dm0) #plot tracks from |0>
        self.dm2bloch(self.dm1) #plot tracks from |1>
        
        self.bloch.xlabel = fig_params['xlabel']
        self.bloch.ylabel = fig_params['ylabel']
        #self.bloch.show()       

        self.bloch.save(data_path+'/'+name+'.pdf')
        
    def dm2bloch(self, dm):
        '''
        add points to the bloch sphere from initial density matrix
        '''
        dm_arr = np.array([dm for i in range(len(self.u_arr))], dtype=qt.Qobj)
        arr = np.zeros((len(self.u_arr), 3))#, dtype=complex)
       
        for i in range(len(self.u_arr)):
            ui = self.u_arr[i]
            if type(ui) != np.ndarray: ui = np2qobj(ui)
            print(ui)
            ui.dims = dm.dims
            dm_arr[i] = ui * dm * ui.dag()
            
            arr[i][0] = (dm_arr[i][1,0] + dm_arr[i][0,1]).real
            arr[i][1] = (1.j * (dm_arr[i][0,1] - dm_arr[i][1,0])).real
            arr[i][2] = (dm_arr[i][0,0] - dm_arr[i][1,1]).real
            
        #self.bloch.add_points(arr1.transpose())
        self.bloch.add_points(arr.transpose(), meth='l')
        #self.bloch.vector_color = [color]
        #self.bloch.point_color = [color]
        #self.bloch.add_vectors(arr2[0])
        #self.bloch.add_vectors(arr2[-1])
        #self.bloch.add_points(arr2[0])
        #self.bloch.add_points(arr2[-1])
        
class plot_bloch_2qb(object):
    def __init__(self):
        self.dm00 = qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,0))) 
        self.dm10 = qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,0))) 
        self.dm01 = qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,1))) 
        self.dm11 = qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,1))) 
        
        self.bloch = qt.Bloch(view=fig_params['view'])
        #self.fig_parms = inp.fig_params_bloch
        
    def plot(self, u_arr, data_path, name='bloch'):
        '''
        u_arr: array of Qobjs of two-qubit-gates
        s0: Qobj of two qubit state
        plot the two qubit states trajectories in one sphere
        '''
        print('Start plotting bloch spheres.')
        self.u_arr = cp.deepcopy(u_arr)
        
        self.dm2bloch(self.dm00) #plot 2qb tracks from |00>
        self.dm2bloch(self.dm10) #plot 2qb tracks from |10>
        
        self.bloch.xlabel = fig_params['xlabel']
        self.bloch.ylabel = fig_params['ylabel']
        #self.bloch.show()       

        self.bloch.save(data_path+'/'+name+'.pdf')
        
    def dm2bloch(self, dm):
        '''
        add points to the bloch sphere from initial density matrix
        '''
        dm_arr = np.array([dm for i in range(len(self.u_arr))], dtype=qt.Qobj)
        dm1_arr = np.array([qt.qeye(2) for i in range(len(self.u_arr))], dtype=qt.Qobj)
        dm2_arr = np.array([qt.qeye(2) for i in range(len(self.u_arr))], dtype=qt.Qobj)
        arr1 = np.zeros((len(self.u_arr), 3))#, dtype=complex)
        arr2 = np.zeros((len(self.u_arr), 3))#, dtype=complex)
        
        for i in range(len(self.u_arr)):
            ui = self.u_arr[i]
            if type(ui) != np.ndarray: ui = np2qobj(ui)
            print(ui)
            ui.dims = dm.dims
            dm_arr[i] = ui * dm * ui.dag()
            
            dm1_arr[i] = dm_arr[i].ptrace(0)
            dm2_arr[i] = dm_arr[i].ptrace(1)
            
            arr1[i][0] = (dm1_arr[i][1,0] + dm1_arr[i][0,1]).real
            arr1[i][1] = (1.j * (dm1_arr[i][0,1] - dm1_arr[i][1,0])).real
            arr1[i][2] = (dm1_arr[i][0,0] - dm1_arr[i][1,1]).real
            
            arr2[i][0] = (dm2_arr[i][1,0] + dm2_arr[i][0,1]).real
            arr2[i][1] = (1.j * (dm2_arr[i][0,1] - dm2_arr[i][1,0])).real
            arr2[i][2] = (dm2_arr[i][0,0] - dm2_arr[i][1,1]).real
            
        self.bloch.add_points(arr1.transpose()) #qb1
        self.bloch.add_points(arr2.transpose(), meth='l') #qb2
        #self.bloch.vector_color = [color]
        #self.bloch.point_color = [color]
        #self.bloch.add_vectors(arr2[0])
        #self.bloch.add_vectors(arr2[-1])
        #self.bloch.add_points(arr2[0])
        #self.bloch.add_points(arr2[-1])
