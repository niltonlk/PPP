import numpy as np
import matplotlib.pyplot as plt

from   scipy.sparse import coo_matrix

#phi function
def phi(u, v_half, slope):
    return 1.0/(1.0+np.exp(-(u-v_half)/slope))

#-----------------------------------------------------------------------------
#function to evaluate the model
#-----------------------------------------------------------------------------
def evaluate(neuron_params, syn_weight, sim_params):
    #initial conditions
    u = np.random.normal(-58.0, 10.0, size=neuron_params['N'])
    phi_u = np.zeros(N)          #array to store phi values

    #array to store spikes
    spk_t = []
    spk_id = []

    trun = 0.0
    while (trun < ttotal):

        #compute phi(T-dt)
        phi_u = phi(u, v_half, slope)

        S = np.sum(phi_u)
        unif = np.random.rand()
        dt = -np.log(unif)/S;

        #compute u(T)
        u = (u-neuron_params['u_reset'])*np.exp(-alpha*dt) + neuron_params['u_reset']

        #compute phi(T)
        phi_u = phi(u, neuron_params['v_half'], neuron_params['slope'])

        unif = np.random.uniform(low=0.0, high=S)

        S_new = np.sum(phi_u)
        trun += dt

        if unif <= S_new:
            phi_cumsum = np.cumsum(phi_u)
            neuron_id = np.where(unif<=phi_cumsum)[0][0]

            u += syn_weight[neuron_id][:]
            u[neuron_id] = neuron_params['u_reset']

            spk_t.append(trun)
            spk_id.append(neuron_id)

    plt.plot(spk_t,spk_id, '.')
    plt.show()

#-----------------------------------------------------------------------------
#parameters
#-----------------------------------------------------------------------------
N = 1000       #network size
v_half = -45.0  #phi function parameter
slope = 2.0     #phi function parameter
tau = 10.0
alpha = 1/tau
u_reset = -65.0 #reset potential
w = 0.15

#dictionary with phi function parameters
params = {'N':N, 'v_half':v_half, 'slope':slope, 'alpha':alpha, 'u_reset':u_reset}

#simulation parameters
ttotal = 1000.0     #total time of simulation in miliseconds
sim_params = {'ttotal': ttotal}

np.random.seed(1000)    #seed for the random number generator

# definition of graph
#-----------------------------------------------------------------------------
#all-to-all:
#-----------------------------------------------------------------------------
#synapses
# syn_weight = np.ones((N,N))*w

#-----------------------------------------------------------------------------
#random network 80% excitatory and 20% inhibitory:
#-----------------------------------------------------------------------------
conn_mat = np.load('graph/brunel_seed_'+str(s)+'.npy', allow_pickle=True).item()

#-----------------------------------------------------------------------------
# running simulation
#-----------------------------------------------------------------------------
evaluate(params, conn_mat.toarray(), sim_params)
