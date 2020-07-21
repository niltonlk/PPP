#############################################################################
# Set Network and neuron parameters
#############################################################################
N = 1000    #number of neurons

r'''
    Network Parameters
    f : Fraction of excitatory neurons
    Nexct : Number of excitatory neurons
    Ninhb : Number of inhibitory neurons
    g     : Excitation/Inhibition ratio
    R     : Ratio between \ni_{thr} and \ni_{ext}
    delay : Synaptic delay
'''
f     = 0.8
g     = 4.0
Nexct = int(f * N)
Ninhb = N-Nexct#int((1-f) * N)
w_ex  = 0.1 # mV
w_in  = -g * w_ex
#delay = 1.5 * ms
Ce    = 1000
Ci    = int( Ce / 4.0 )

r'''
	All-to-all
'''
w = 0.01

r'''
	Simulation Parameters
'''
s    = 1
Tsim = 100.0
#simulation parameters
sim_params = {'ttotal': Tsim}

r'''
  Neuron Parameters
  tau_m : Membrane time constant
  tau_ref : Refractory period
  Vrest   : Resting potential
  Vreset  : Reset potential
  Vth     : Threshold
  eqs     : Model equation
  v_ext   : External input frequency
 '''
# tau      = 0.020 # ms (20)
# tau_syn  = 0.010
#tau_ref  =  2. # ms
#Vrest    =  0. # mV
#Vreset   = 10. # mV
#Vth      = 20. # mV

r'''
	Phi parameters
'''
# v_half  = 20.0   #phi function parameter
# slope   = 1.0      #phi function parameter
# alpha   = 1.0/tau
# beta    = 1.0/tau_syn
# u_rest  = 0
# u_reset = 0.0 #reset potential



#############################################################################
# Neuron parameters
#############################################################################
tau_m       = 10.0  # membrane time constant in (ms)
t_ref       = 2.0   # refractory period in (ms)
V_reset     = 0.0   # reset membrane potential in (mV)
V_rest      = 0.0   # resting membrane potential (mV)

#############################################################################
# Synapse parameters
#############################################################################
tau_syn_ex  = 0.5 # excitatory synaptic time constant in (ms)
tau_syn_in  = 0.5 # inhibitory synaptic time constant in (ms)
beta_ex     = 1.0/tau_syn_ex
beta_in     = 1.0/tau_syn_in
beta        = beta_ex

I_ext       = 100.0 # constant current in (pA)

#############################################################################
# Phi parameters
#############################################################################
alpha       = 1.0/tau_m
gamma       = 0.1           # slope of the firing probability funtion in (1/mV)
r           = 0.4           # curvature of the firing probability function (unitless)
V_rheo      = 15.0          # rheobase potential, potential in which firing probability becomes > 0 in (mV)

#############################################################################
# Poisson background parameters
#############################################################################
poisson_rate = 8.0*10 # rate of poisson spike input in (Hz)
poisson_weight = 87.8 # weight of poisson input in (pA)
