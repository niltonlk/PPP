#!/usr/bin/env pypy

import numpy             as np
import matplotlib.pyplot as plt
from   set_params        import *
import time

#phi function
'''def phi(V, v_half, slope):
    #return 1.0/(1.0+np.exp(-(V-v_half)/slope))
    # return (1/27.07) * np.exp((V-v_half)/slope) + 1e-4
    phi_u = slope*V#+0.1
    phi_u[phi_u>1] = 1
    phi_u[phi_u<0] = 0
    return phi_u'''

def phi(V, gamma, r):
    V_diff = V - V_rheo
    idx_neg = np.where(V_diff < 0)
    V_diff[idx_neg] = 0.0
    phi_u = (np.power(gamma * V_diff, r))/100.0#*10.0
    # Phi in kHz - divided by 0.1
    # phi_u[phi_u<0] = 0
    return phi_u

#-----------------------------------------------------------------------------
#function to evaluate the model
#-----------------------------------------------------------------------------
def evaluate(post_list):
    #initial conditions
    V = np.random.uniform(0.0, V_rheo+1.0, size=N )
    phi_u = np.zeros(N)          #array to store phi values
    I_syn = np.zeros(N)          #array to store synaptic current values
    last_spike = np.zeros(N)

    #array to store spikes
    spk_t = []
    spk_id = []

    trun = 0.0
    while (trun < t_sim):
        #compute phi(T-dt)
        phi_u = phi(V, gamma, r)
        S = np.sum(phi_u)
        unif = np.random.rand()
        dt = -np.log(unif)/S;

        # compute I
        I_syn = I_syn*np.exp(-beta*dt)
        #compute V(T)
        V = (V-V_rest)*np.exp(-alpha*dt) + V_rest + I_ext + I_syn

        #compute phi(T)
        phi_u = phi(V, gamma, r)

        unif = np.random.uniform(low=0.0, high=S)

        S_new = np.sum(phi_u)
        trun += dt

        if unif<=S_new:
            phi_cumsum = np.cumsum(phi_u)
            neuron_id = np.where(unif<=phi_cumsum)[0][0]

            # checking refractory period
            if last_spike[neuron_id]==0 or (trun-last_spike[neuron_id])>=delay:

                # updating of postsynaptic currents:
                I_syn[post_list[neuron_id][0]] += post_list[neuron_id][1]

                # updating of postsynaptic potentials:
                V[post_list[neuron_id][0]] += post_list[neuron_id][1]

                # updating of last spike list:
                last_spike[neuron_id] = trun

                # recording spike time and neuron index:
                spk_t.append(trun)
                spk_id.append(neuron_id)

            V[neuron_id] = V_reset

    print(len(spk_t)/N)

    return np.array(spk_t), np.array(spk_id)

#-----------------------------------------------------------------------------
#parameters
#-----------------------------------------------------------------------------
np.random.seed(rseed)    #seed for the random number generator

#-----------------------------------------------------------------------------
#random network 80% excitatory and 20% inhibitory:
#-----------------------------------------------------------------------------
from generate_graph import *
post_list = brunel_graph(N, w_ex, g, save_graph=False)

# post_list = all_to_all(N, w)

#-----------------------------------------------------------------------------
#running simulation
#-----------------------------------------------------------------------------
init = time.time()
spk_t, spk_id = evaluate(post_list)
end  = time.time()
print('Simulation time:' + str(end-init))

#-----------------------------------------------------------------------------
#plot graph
#-----------------------------------------------------------------------------
# plt.plot(spk_t, spk_id, '.k', markersize=1.0)
plt.plot(spk_t[spk_id<=N*0.8],spk_id[spk_id<=N*0.8], '.k', markersize=1.0)
plt.plot(spk_t[spk_id>N*0.8],spk_id[spk_id>N*0.8], '.r', markersize=1.0)
plt.tight_layout()
plt.show()
# plt.savefig('array.png', dpi = 600)
# plt.close()
