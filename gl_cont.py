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
Nexc = int(N*0.8)   #number of excitatory neurons
Nin = int(N-Nexc)   #number of inhibitory neurons

p = 0.01                    #probability of connection
Nconn_exc = int(p*Nexc*N)   #number of connections from excitatory neurons to all
Nconn_in = int(p*Nin*N)     #number of connections from inhibitory neurons to all

#list of pre (excitatory) and post-synaptic neurons connected; and list of synaptic weights
pre_list_exc  = np.random.randint(low=0.0, high=N-1, size=Nconn_exc)
post_list_exc  = np.random.randint(low=0.0, high=N-1, size=Nconn_exc)
syn_weight_exc = np.ones(Nconn_exc)*w

#list of pre (inhibitory) and post-synaptic neurons connected; and list of synaptic weights
pre_list_in  = np.random.randint(low=0.0, high=N-1, size=Nconn_in)
post_list_in  = np.random.randint(low=0.0, high=N-1, size=Nconn_in)
syn_weight_in = -np.ones(Nconn_in)*w

#concatenation of the lists defined above
pre_list = np.concatenate((pre_list_exc, pre_list_in))
post_list = np.concatenate((post_list_exc, post_list_in))
syn_weight_list = np.concatenate((syn_weight_exc, syn_weight_in))

#syn_weight is the matrix with all connections and synaptic weights
syn_weight = coo_matrix((syn_weight_list, (pre_list, post_list)), shape=(N, N)).toarray()
del pre_list, pre_list_exc, pre_list_in, post_list_exc, post_list_in, \
    syn_weight_list, syn_weight_exc, syn_weight_in

#-----------------------------------------------------------------------------
# running simulation
#-----------------------------------------------------------------------------
evaluate(params, syn_weight, sim_params)
