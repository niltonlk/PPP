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
    phi_u = np.zeros(N)         # array to store phi values
    I_syn = np.zeros(N)         # array to store synaptic current values
    last_spike = np.zeros(N)    # array to store the last spike for each neuron
    next_spike = np.zeros(N)    # array to store the possible next spike for each neuron

    #array to store spikes
    spk_t = []
    spk_id = []

    #initial conditions
    trun = 0.0
    V = np.random.uniform(0.0, V_rheo+1.0, size=N )
    phi_u = phi(V, gamma, r)
    next_spike = trun + np.random.rand(N)/phi_u #WRONG

    while (trun < t_sim):

        # find next spike
        id_min = np.argmin(next_spike)
        dt = next_spike[id_min]

        # update clock time
        trun += dt

        # record spike time and neuron index:
        spk_t.append(trun)
        spk_id.append(id_min)

        # time spent since last spike
        delta_spk = trun - last_spike[post_list[id_min][0]]

        #compute V(T)
        V[post_list[id_min][0]] = (V[post_list[id_min][0]]-V_rest)*np.exp(-alpha*delta_spk) \
            + I_syn[post_list[id_min][0]]*np.exp(-beta*delta_spk)*(np.exp((beta-alpha)*delta_spk)-1)/(beta-alpha) \
            + I_ext*np.exp(-beta*delta_spk)*(np.exp((beta)*delta_spk)-1)/(beta)

        # compute I at time T
        I_syn[post_list[id_min][0]] = I_syn[post_list[id_min][0]]*np.exp(-beta*dt) + post_list[id_min][1]

        # reset V(T) of neuron who spiked
        V[id_min] = V_reset

        # update next spike of the actual neuron who is spiking
        phi_u[id_min] = 0.0
        next_spike[id_min] = trun + np.random.rand()/phi_u[id_min] #WRONG

        #compute phi(V) at time T
        phi_u[post_list[id_min][0]] = phi(V[post_list[id_min][0]], gamma, r)
        next_spike[post_list[id_min][0]] = trun + np.random.rand(len(post_list[id_min][0]))/phi_u[post_list[id_min][0]] #WRONG

    print('\nNumber of spikes per neuron: ' + str(len(spk_t)/N))

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
