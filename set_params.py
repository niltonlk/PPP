#############################################################################
# Set Network and neuron parameters
#############################################################################

r'''
    Network Parameters
    N : Number of neurons in the Network
    f : Fraction of excitatory neurons
    Nexct : Number of excitatory neurons
    Ninhb : Number of inhibitory neurons
    g     : Excitation/Inhibition ratio
    R     : Ratio between \ni_{thr} and \ni_{ext}
    delay : Synaptic delay
'''
f     = 0.8
g     = 4.0
N     = 12500
Nexct = int(f * N)
Ninhb = int((1-f) * N)
w_ex  = 0.1 # mV
w_in  = -g * w_ex
#delay = 1.5 * ms
Ce    = 1000
Ci    = int( Ce / 4.0 )

r'''
	Simulation Parameters
'''
s    = 1
Tsim = 1000
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
tau      = 20. # ms (20)
#tau_ref  =  2. # ms
#Vrest    =  0. # mV
#Vreset   = 10. # mV
#Vth      = 20. # mV

r'''
	Phi parameters
'''
v_half  = 20.0   #phi function parameter
slope   = 1.2      #phi function parameter
alpha   = 1/tau
u_rest  = 0
u_reset = 10.0 #reset potential

#dictionary with phi function parameters
params = {'N':N, 'v_half':v_half, 'slope':slope, 'alpha':alpha, 'u_reset':u_reset, 'u_rest': u_rest}
