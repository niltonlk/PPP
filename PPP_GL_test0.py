#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:19:14 2019

@author: nilton
"""

import numpy as np
#import mpi4py
#import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd

# Define Network Parameters extracted from original Potjans and Diesmann model
net_dict = {
    # Names of the simulated populations.
    'populations': ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I'],
    # Number of neurons in the different populations. The order of the
    # elements corresponds to the names of the variable 'populations'.
    'N_full': np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]),
    # Mean rates of the different populations in the non-scaled version
    # of the microcircuit. Necessary for the scaling of the network.
    # The order corresponds to the order in 'populations'.
    'full_mean_rates':
        np.array([0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]),
    # Connection probabilities. The first index corresponds to the targets
    # and the second to the sources.
    'conn_probs':
        np.array(
            [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0., 0.0076, 0.],
             [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0., 0.0042, 0.],
             [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0., 0.1057, 0.],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443]]),
    # Number of external connections to the different populations.
    # The order corresponds to the order in 'populations'.
    'K_ext': np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]),
    # Factor to scale the indegrees.
    'K_scaling': 1.0,
    # Factor to scale the number of neurons.
    'N_scaling': 1.0,
    # Mean amplitude of excitatory postsynaptic potential (in mV).
    # The order corresponds to the order in 'populations'.
    'PSP_e':
        np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]),
    # Relative standard deviation of the postsynaptic potential.
    # The order corresponds to the order in 'populations'.
    'PSP_sd':
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    # Relative inhibitory synaptic strength (in relative units).
    'g': -4,
    # Rate of the Poissonian spike generator (in Hz).
    # The order corresponds to the order in 'populations'.
    'bg_rate':
        np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
    # Turn Poisson input on or off (True or False).
    'poisson_input': True,
    # Delay of the Poisson generator (in ms).
    'poisson_delay': 1.5,
    # Mean delay of excitatory connections (in ms).
    'mean_delay_exc': 1.5,
    # Mean delay of inhibitory connections (in ms).
    'mean_delay_inh': 0.75,
    # Relative standard deviation of the delay of excitatory and
    # inhibitory connections (in relative units).
    'rel_std_delay': 0.5,
    # Parameters of the neurons.
    'neuron_params': { # L23e
        # Membrane potential average for the neurons (in mV).
        'V0_mean': -58.0,
        # Standard deviation of the average membrane potential (in mV).
        'V0_sd': 10.0,
        # Reset membrane potential of the neurons (in mV).
        'E_L': -65.0,
        # Threshold potential of the neurons (in mV).
        'V_th': -50.0,
        # Membrane potential after a spike (in mV).
        'V_reset': -65.0,
        # Membrane capacitance (in pF).
        'C_m': 250.0,
        # Membrane time constant (in ms).
        'tau_m': 10.0,
        # Time constant of postsynaptic excitatory currents (in ms).
        'tau_syn_ex': 0.5,
        # Time constant of postsynaptic inhibitory currents (in ms).
        'tau_syn_in': 0.5,
        # Time constant of external postsynaptic excitatory current (in ms).
        'tau_syn_E': 0.5,
        # Refractory period of the neurons after a spike (in ms).
        't_ref': 2.0},
    }

# Phi function
def Phi(v, v_half=-45., slope=2.): 
    return 1/(1 + np.exp(-(v-v_half)/slope))

# Total number of neurons in the network
N = 200

# Seed the random number generator
np.random.seed(1000)

# Create N u neurons (membrane potential) with initial conditions with a normal distribution
# with mean V0_mean and standard deviation V0_sd
u = net_dict['neuron_params']['V0_sd']*np.random.randn(N) + net_dict['neuron_params']['V0_mean']

# Create connection
# all to all

# Simulation time in miliseconds
t_end = 1.0e3

# Synaptic weight [mV]
W_e = 0.15  # excitatory
g = 4  # inhibitory/excitatory ration
W_i = -g*W_e  # inhibitory

t = 0.0

V_reset = net_dict['neuron_params']['V_reset']

#st = pd.DataFrame(columns=['i', 't', 'V_m']).astype({'i':'int', 't':'float', 'V_m':'float'})

idx = []
t_spk = []

while(t < t_end):
    # Calculate all Phi(v)
    PHI = Phi(u)
    # Calculate sum of all Phi(v)
    S_PHI = np.sum(PHI)
    # Calculate cumulative sum of all Phi(v)
    CS_PHI = np.cumsum(PHI)
    # Calculate dt
    dt = -np.log(np.random.rand())/S_PHI
    # Calculate leak rate
    du = np.exp(-dt/net_dict['neuron_params']['tau_m'])
    # Update u for all neurons
    u = du*(u - net_dict['neuron_params']['V_reset']) + net_dict['neuron_params']['V_reset']
    # Draw a uniform number to determine which neuron fires
    s = np.random.uniform(0.0, S_PHI)
    # Calculate updated values of Phi(v)
    NEW_PHI = Phi(u)
    S_NEW_PHI = np.sum(NEW_PHI)
    CS_NEW_PHI = np.cumsum(NEW_PHI)

    # update time
    t += dt

    # Condition where a neuron fired (except neuron 0)
    if s < CS_NEW_PHI[-1]:
        # Determine neuron 
        i = np.where(CS_NEW_PHI > s)[0][0]
        # Store spike time 
#        st = st.append({'i':i, 't':t, 'V_m':u[i]}, ignore_index=True).astype({'i':'int'})
        idx.append(i)
        t_spk.append(t)
        # Increment all u by W_e
        u += W_e
        # Reset u of the neuron that fired
        u[i] = net_dict['neuron_params']['V_reset']
    # Condition where no neuron fired
#    else:
#        i = np.NaN
    

#    if not(np.isnan(i)):
#    if i:
#        print(t, i, u[i])
#        # Increment all u by W_e
#        u += W_e
#        # Reset u of the neuron that fired
#        u[i] = net_dict['neuron_params']['V_reset']
    

plt.plot(t_spk, idx, ' .')