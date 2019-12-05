import numpy as np
import matplotlib.pyplot as plt

#function
def phi(u, v_half, slope):
    return 1.0/(1.0+np.exp(-(u-v_half)/slope))

#parameters
N = 200  #network size
v_half = -45.0 #phi function parameter
slope = 2.0 #phi function parameter
tau = 10.0
alpha = 1/tau
u_reset = -65.0 #reset potential
# w = 0.15
w = 0.15

#simulation parameters
ttotal = 1000.0
trun = 0.0

np.random.seed(1000)

#synapses
syn_pre = np.array([0,1])
syn_post = np.array([1,0])

#initial conditions
u = np.zeros(N)         #initializing membrane potentials
u = np.random.normal(-58.0, 10.0, size=N)
phi_u = np.zeros(N)          #array to store phi values

#array to store spikes
spk_t = []
spk_id = []

while (trun < ttotal):

    #compute phi(T-dt)
    phi_u = phi(u, v_half, slope)

    S = np.sum(phi_u)
    unif = np.random.rand()
    dt = -np.log(unif)/S;

    #compute u(T)
    u = (u-u_reset)*np.exp(-alpha*dt) + u_reset

    #compute phi(T)
    phi_u = phi(u, v_half, slope)

    unif = np.random.uniform(low=0.0, high=S)

    S_new = np.sum(phi_u)
    trun += dt

    if unif <= S_new:
        phi_cumsum = np.cumsum(phi_u)
        neuron_id = np.where(unif<=phi_cumsum)[0][0]

        u += w
        u[neuron_id] = u_reset

        spk_t.append(trun)
        spk_id.append(neuron_id)

plt.plot(spk_t,spk_id, '.')
plt.show()
