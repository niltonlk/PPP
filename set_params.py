#############################################################################
# Simulation parameters
#############################################################################
rseed = 1           # random generator seed
t_sim  = 500.0         # simulation time

# sim_params = {'t_sim': t_sim}

#############################################################################
# Network parameters
#############################################################################
N     = 1000            # number of neurons
g     = 6.0             # inhibition/excitation ratio

frac  = 80000.0/N**0.5  # reescale factor

w_ex  = frac*0.15       # excitatory synaptic weight in (mV)
w_in  = -g * w_ex       # inhibitory synaptic weight in (mV)

d_ex  = 1.5             # excitatory synaptic delay in (ms)
d_in  = 0.8             # inhibitory synaptic delay in (ms)

#############################################################################
# Neuron parameters
#############################################################################
tau_m       = 10.0      # membrane time constant in (ms)
alpha       = 1.0/tau_m # inverse of membrane time constant in (1/ms)
t_ref       = 2.0       # refractory period in (ms)
V_reset     = 0.0       # reset membrane potential in (mV)
V_rest      = 0.0       # resting membrane potential (mV)

#############################################################################
# Phi parameters
#############################################################################
gamma       = 0.1           # slope of the firing probability funtion in (1/mV)
r           = 0.4           # curvature of the firing probability function (unitless)
V_rheo      = 15.0          # rheobase potential, potential in which firing probability becomes > 0 in (mV)

#############################################################################
# Synapse parameters
#############################################################################
tau_syn_ex  = 0.5               # excitatory synaptic time constant in (ms)
tau_syn_in  = 0.5               # inhibitory synaptic time constant in (ms)
beta_ex     = 1.0/tau_syn_ex
beta_in     = 1.0/tau_syn_in
beta        = beta_ex

#############################################################################
# External input parameters
#############################################################################
I_ext       = 10.0 # constant current in (pA)

#############################################################################
# Poisson background parameters
#############################################################################
poisson_rate   = 8.0*10.0 # rate of poisson spike input in (Hz)
poisson_weight = 87.8 # weight of poisson input in (pA)
