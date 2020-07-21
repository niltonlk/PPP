import numpy             as np
import matplotlib.pyplot as plt
from   set_params        import *


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
    phi_u = (np.power(gamma * V_diff, r))*10.0
    # Phi in kHz - divided by 0.1
    # phi_u[phi_u<0] = 0
    return phi_u

#-----------------------------------------------------------------------------
#function to evaluate the model
#-----------------------------------------------------------------------------
def evaluate(syn_weight, sim_params):
    #initial conditions
    V = np.random.uniform(0.0, V_rheo+1.0, size=N )
    phi_u = np.zeros(N)          #array to store phi values
    I_syn = np.zeros(N)          #array to store synaptic current values

    #array to store spikes
    spk_t = []
    spk_id = []

    trun = 0.0
    while (trun < Tsim):
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

        if unif <= S_new:
            print(trun)
            phi_cumsum = np.cumsum(phi_u)
            neuron_id = np.where(unif<=phi_cumsum)[0][0]

            I_syn += syn_weight[neuron_id][:]
            V += syn_weight[neuron_id][:]
            V[neuron_id] = V_reset

            spk_t.append(trun)
            spk_id.append(neuron_id)

    print(len(spk_t)/N)

    return np.array(spk_t), np.array(spk_id)

#-----------------------------------------------------------------------------
#parameters
#-----------------------------------------------------------------------------
np.random.seed(s)    #seed for the random number generator

#-----------------------------------------------------------------------------
#random network 80% excitatory and 20% inhibitory:
#-----------------------------------------------------------------------------
from generate_graph import *
# syn_weight = brunel_graph(Ce, Ci, Nexct, Ninhb, w_ex, g, save_graph=False)

syn_weight = all_to_all(N, w)

#-----------------------------------------------------------------------------
#running simulation
#-----------------------------------------------------------------------------
spk_t, spk_id = evaluate(syn_weight, sim_params)

#-----------------------------------------------------------------------------
#plot graph
#-----------------------------------------------------------------------------
plt.plot(spk_t, spk_id, '.k', markersize=1.0)
# plt.plot(spk_t[spk_id<=10000],spk_id[spk_id<=10000], '.k', markersize=1.0)
# plt.plot(spk_t[spk_id>10000],spk_id[spk_id>10000], '.r', markersize=1.0)
plt.tight_layout()
plt.show()
# plt.savefig('array.png', dpi = 600)
# plt.close()
