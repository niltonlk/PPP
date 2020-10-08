import numpy             as np
import matplotlib.pyplot as plt
from   set_params        import *
import time

#############################################################################
# Phi function
#############################################################################
def phi(V, gamma, r):
    V_diff = V - V_rheo
    idx_neg = np.where(V_diff < 0)[0]
    V_diff[idx_neg] = 0.0
    phi_u = (np.power(gamma * V_diff, r))
    # idx_1 = np.where(phi_u>1)[0]
    return phi_u*0.01

#############################################################################
# Function to evaluate the model
#############################################################################
def evaluate(post_list):
    #initial conditions
    V = np.random.uniform(0.0, V_rheo+1.0, size=N )
    phi_u      = np.zeros(N)          # array to store phi values
    I_syn      = np.zeros(N)          # array to store synaptic current values
    last_spike = np.zeros(N)          # array to store last spikes for each neuron

    # arrays to store data
    spk_t = []  # spiking times
    spk_id = [] # index of the spiking neurons

    # spk_buffer stores the spikes that not arrived in the postsynaptic neurons yet
    # Once it is considered, the respective spike is removed from this list
    # Each row of this variable stores (target neuron, synaptic weigth, spiking time + delay)
    spk_buffer = np.zeros((1,3))       # initializing with zeros

    trun = 0.0  # initial time of simulation

    size_buffer = []

    # The simulation will run until trun < total time of simulation.
    # If the sum of all Phi is zero, then the simulation will stop too.
    while (trun < t_sim):

        # compute spikes arrived from (T-dt) on
        spk_buffer = spk_buffer[np.argsort(spk_buffer[:,2])] # sort by spikes arrived first
        idx_del= np.where((spk_buffer[:,2]>0.0))[0] # find the index of spikes ocurred in T-dt and t_sim (end of simulation)

        size_buffer.append(len(spk_buffer))

        if len(idx_del)>0:
            S_aux = []
            trun_ = trun
            V_ = V
            I_syn_ = I_syn
            # compute spikes between T-dt and t_sim (end of simulation)
            for id in idx_del:
                #compute phi(V) at time (T-dt)
                phi_u = phi(V_+I_syn_/(beta-alpha), gamma, r)
                # phi_u = phi(V, gamma, r)

                S = np.sum(phi_u)           #sum of the rates
                S_aux.append(S)

                dt_ = spk_buffer[id,2]-trun_

                # compute V(T)
                V_ = (V_-V_rest)*np.exp(-alpha*dt_) \
                    + I_syn_*np.exp(-beta*dt_)*(np.exp((beta-alpha)*dt_)-1)/(beta-alpha) \
                    + I_ext*np.exp(-beta*dt_)*(np.exp((beta)*dt_)-1)/(beta)

                # compute I(T)
                I_syn_ = I_syn*np.exp(-beta*dt_)

                # update postsynaptic current
                I_syn_[spk_buffer[id,0].astype(int)] += spk_buffer[id,1]*np.exp(-beta*(dt_))

                trun_ += dt_

            S = np.max(S_aux)

        else:
            phi_u = phi(V, gamma, r)
            S = np.sum(phi_u)           #sum of the rates

            if S==0.0: break

        # roll an uniform
        unif = np.random.rand()
        # find the next spiking time
        dt   = -np.log(unif)/S;

        # update running time
        trun += dt

        # compute V(T)
        V = (V-V_rest)*np.exp(-alpha*dt) \
            + I_syn*np.exp(-beta*dt)*(np.exp((beta-alpha)*dt)-1)/(beta-alpha) \
            + I_ext*np.exp(-beta*dt)*(np.exp((beta)*dt)-1)/(beta)

        # compute I(T)
        I_syn = I_syn*np.exp(-beta*dt)

        # compute spikes arrived between T-dt and T
        # spk_buffer = spk_buffer[np.argsort(spk_buffer[:,2])] # sort by spikes arrived first
        idx_del= np.where((spk_buffer[:,2]<=trun)&(spk_buffer[:,2]>0.0))[0] # find the index of spikes ocurred in T-dt and T
        spk_dt = spk_buffer[idx_del,:]  # list of spikes in the time window T-dt to T

        # if there is no spike between T-dt and T then update I_syn normally
        # compute all spikes that ocurred in this time window and update I_syn accordingly
        # compute spikes between T-dt and T
        for id in idx_del:
            I_syn[spk_buffer[id,0].astype(int)] += spk_buffer[id,1]*np.exp(-beta*(trun-spk_buffer[id,2]))
            V[spk_buffer[id,0].astype(int)]     += spk_buffer[id,1]*\
                (np.exp(-alpha*(trun-spk_buffer[id,2]))-np.exp(-beta*(trun-spk_buffer[id,2])))/\
                (beta-alpha)
        spk_buffer = np.delete(spk_buffer, obj=idx_del, axis=0)

        #compute phi(V) at time T
        phi_u = phi(V, gamma, r)
        S_new = np.sum(phi_u) # sum of the rates

        unif = np.random.uniform(low=0.0, high=S)

        if unif<=S_new:
            phi_cumsum = np.cumsum(phi_u)
            neuron_id = np.where(unif<=phi_cumsum)[0][0]

            # checking refractory period
            if last_spike[neuron_id]==0 or (trun-last_spike[neuron_id])>=t_ref:

                # updating of last spike list:
                last_spike[neuron_id] = trun

                # updating delayed spike buffer
                tmp_array  = np.c_[post_list[neuron_id][0], post_list[neuron_id][1], trun+post_list[neuron_id][2]]
                spk_buffer = np.r_[spk_buffer, tmp_array]

                # recording spike time and neuron index:
                spk_t.append(trun)
                spk_id.append(neuron_id)

            V[neuron_id] = V_reset

    print('\nNumber of spikes per neuron: ' + str(len(spk_t)/N))

    return np.array(spk_t), np.array(spk_id), size_buffer

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
post_list = brunel_graph(N, w_ex, g, d_ex, d_in, save_graph=False)
end  = time.time()
print('...time spent: ' + str(end-init))

#-----------------------------------------------------------------------------
#running simulation
#-----------------------------------------------------------------------------
print('\nRunning the simulation...')
init = time.time()
spk_t, spk_id, size_buffer = evaluate(post_list)
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
