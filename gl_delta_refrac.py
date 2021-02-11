import numpy             as np
import matplotlib.pyplot as plt
from   set_params        import *
import time

#phi function
def phi(V, gamma, r):
    V_diff = V - V_rheo
    idx_neg = np.where(V_diff < 0)
    V_diff[idx_neg] = 0.0
    phi_u = (np.power(gamma * V_diff, r))/100.0#*10.0
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
        phi_u = phi(V, gamma, r) # it works with delta synapses
        S = np.sum(phi_u)
        if S == 0: break
        unif = np.random.rand()
        dt = -np.log(unif)/S;

        #compute V(T)
        V = (V-V_rest)*np.exp(-alpha*dt) + V_rest + I_ext + I_syn

        #compute phi(V) at time T
        phi_u = phi(V, gamma, r) # it works with delta synapses

        unif = np.random.uniform(low=0.0, high=S)

        S_new = np.sum(phi_u)
        trun += dt

        if unif<=S_new:
            phi_cumsum = np.cumsum(phi_u)
            neuron_id = np.where(unif<=phi_cumsum)[0][0]

            # checking refractory period
            if last_spike[neuron_id]==0 or (trun-last_spike[neuron_id])>=t_ref:

                # updating of postsynaptic currents:
                V[post_list[neuron_id][0]] += post_list[neuron_id][1]

                # updating of last spike list:
                last_spike[neuron_id] = trun

                # recording spike time and neuron index:
                spk_t.append(trun)
                spk_id.append(neuron_id)

            V[neuron_id] = V_reset

    print('\nMean firing rate in Hz: ' + str(len(spk_t)/N/(t_sim/1000.0)))

    return np.array(spk_t), np.array(spk_id)

#-----------------------------------------------------------------------------
#parameters
#-----------------------------------------------------------------------------
np.random.seed(rseed)    #seed for the random number generator

#-----------------------------------------------------------------------------
#random network 80% excitatory and 20% inhibitory:
#-----------------------------------------------------------------------------
from generate_graph import *
print('\nBuilding graph...')
init = time.time()
post_list = brunel_graph(N, w_ex, g, save_graph=False)
end  = time.time()
print('...time spent: ' + str(end-init))

#-----------------------------------------------------------------------------
#running simulation
#-----------------------------------------------------------------------------
print('\nRunning the simulation...')
init = time.time()
spk_t, spk_id = evaluate(post_list)
end  = time.time()
print('\nSimulation time: ' + str(end-init))

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
