import numpy             as np
import matplotlib.pyplot as plt
from   scipy.sparse      import coo_matrix
from   set_params        import *

#phi function
def phi(u, v_half, slope):
    return 1.0/(1.0+np.exp(-(u-v_half)/slope))

#-----------------------------------------------------------------------------
#function to evaluate the model
#-----------------------------------------------------------------------------
def evaluate(neuron_params, syn_weight, sim_params):
    #initial conditions
    u = np.random.normal(-58.0, 10.0, size=neuron_params['N'] )
    phi_u = np.zeros(N)          #array to store phi values

    #array to store spikes
    spk_t = []
    spk_id = []

    trun = 0.0
    while (trun < Tsim):

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
    #plt.show()

#-----------------------------------------------------------------------------
#parameters
#-----------------------------------------------------------------------------
np.random.seed(s)    #seed for the random number generator

#-----------------------------------------------------------------------------
#random network 80% excitatory and 20% inhibitory:
#-----------------------------------------------------------------------------
conn_mat = np.load('graph/brunel_seed_'+str(1)+'.npy', allow_pickle=True).item()

#-----------------------------------------------------------------------------
# running simulation
#-----------------------------------------------------------------------------
evaluate(params, conn_mat.toarray(), sim_params)
plt.savefig('semarray.png', dpi = 600)
plt.close()