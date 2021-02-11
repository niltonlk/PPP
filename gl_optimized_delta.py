import numpy             as np
import matplotlib.pyplot as plt
from   set_params        import *
import time

#phi function
def phi(V, gamma, r):
    V_diff = V - V_rheo
    idx_neg = np.where(V_diff < 0)
    V_diff[idx_neg] = 0.0
    phi_u = (np.power(gamma * V_diff, r))/100.0
    return phi_u

def phi_single(V, gamma, r):
    V_diff = V - V_rheo
    if V_diff<0: V_diff = 0.0
    phi_u = (np.power(gamma * V_diff, r))/100.0
    return phi_u

#-----------------------------------------------------------------------------
#function to evaluate the model
#-----------------------------------------------------------------------------
def evaluate(post_list):
    #initial conditions
    V = np.random.uniform(0.0, V_rheo+1.0, size=N )
    phi_u = np.zeros(N)         # array to store phi values
    I_syn = np.zeros(N)         # array to store synaptic current values
    last_spike  = np.zeros(N)    # array to store the last spike for each neuron
    next_spike  = np.zeros(N)    # array to store the possible next spike for each neuron
    last_update = np.zeros(N)    # array to store the last update for each neuron

    #array to store spikes
    spk_t = []
    spk_id = []

    #initial conditions
    trun = 0.0
    V = np.random.uniform(0.0, V_rheo+1.0, size=N )
    phi_u = phi(V, gamma, r)

    # starting all next spikes as infinite (ending time of simulation)
    next_spike = np.ones(N)*t_sim

    # play uniform to find next spikes
    nonzero_phi_id = np.nonzero(phi_u)
    next_spike[nonzero_phi_id] = trun - np.log(np.random.rand(len(nonzero_phi_id)))/phi_u[nonzero_phi_id]

    while (trun < t_sim):

        # find next spike time and neuron index
        id_min = np.argmin(next_spike)
        dt = next_spike[id_min] - last_update[id_min]

        # compute phi(T-dt) of spiking neuron candidate (neuron i) and play uniform
        unif = np.random.uniform(low=0.0, high=phi_u[id_min])

        # update clock time
        trun = next_spike[id_min]

        # save the update time for neuron i
        last_update[id_min] = trun

        #compute V(T) of neuron i
        V[id_min] = (V[id_min]-V_rest)*np.exp(-alpha*dt) + V_rest + I_ext

        # compute phi(V) at time T of neuron i
        phi_u[id_min] = phi_single(V[id_min], gamma, r)   # it works with delta synapses

        if unif <= phi_u[id_min]:
            # record spike time and neuron index:
            spk_t.append(trun)
            spk_id.append(id_min)

            # time spent since last update
            dt = trun - last_update[post_list[id_min][0]]

            #compute V(T) of receiving neurons plus the synaptic weight coming from the presynaptic neuron
            V[post_list[id_min][0]] = (V[post_list[id_min][0]]-V_rest)*np.exp(-alpha*dt) + V_rest + I_ext + post_list[id_min][1]

            # reset V(T) of neuron who spiked
            V[id_min] = V_reset

            # update phi(V) at time T of neuron who spiked
            phi_u[id_min] = phi_single(V[id_min], gamma, r) # it works for delta synapses

            # update next spike of the actual neuron who is spiking
            if phi_u[id_min]>0.0:
                next_spike[id_min] = trun - np.log(np.random.rand())/phi_u[id_min]
            else:
                next_spike[id_min] = t_sim

            # compute phi(V) at time T of the receiving neurons
            phi_u[post_list[id_min][0]] = phi(V[post_list[id_min][0]], gamma, r)  # it works for delta synapses

            # update next spike of target neurons
            nonzero_phi_id = post_list[id_min][0][phi_u[post_list[id_min][0]]>0.0]  # nonzero elements index
            zero_phi_id    = post_list[id_min][0][phi_u[post_list[id_min][0]]==0.0] # zero elements index
            next_spike[zero_phi_id]    = t_sim  # neuron with rate equals zero will spike at infinity (t_sim for the simulation)
            next_spike[nonzero_phi_id] = trun - np.log(np.random.rand(len(nonzero_phi_id)))/phi_u[nonzero_phi_id]

            # save the update time for neuron i
            last_update[post_list[id_min][0]] = trun

        else:
            # update next spike of neuron i
            if phi_u[id_min]>0.0:
                next_spike[id_min] = trun - np.log(np.random.rand())/phi_u[id_min]
            else:
                next_spike[id_min] = t_sim

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
