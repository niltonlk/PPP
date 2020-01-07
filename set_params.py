#############################################################################
# Set Network and neuron parameters
#############################################################################
N = 1000    #number of neurons

Iext = 0.01 #external constant current

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
w = 0.04

r'''
	Simulation Parameters
'''
s    = 1
Tsim = 10.0
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
tau      = 0.020 # ms (20)
#tau_ref  =  2. # ms
#Vrest    =  0. # mV
#Vreset   = 10. # mV
#Vth      = 20. # mV

r'''
	Phi parameters
'''
v_half  = 20.0   #phi function parameter
slope   = 1.0      #phi function parameter
alpha   = 1.0/tau
u_rest  = 0
u_reset = 0.0 #reset potential

#dictionary with phi function parameters
params = {'N':N, 'v_half':v_half, 'slope':slope, 'alpha':alpha, 'u_reset':u_reset, 'u_rest': u_rest}
