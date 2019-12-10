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
g     = 6.0
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
Tsim = 5000

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
tau_m    = 20. # ms
#tau_ref  =  2. # ms
Vrest    =  0. # mV
Vreset   = 10. # mV
Vth      = 20. # mV