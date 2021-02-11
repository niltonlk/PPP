# PPP
Simulation of a Poisson Point Process using the exact simulation algorithm by Gillespie, 2007


## Code repository

*  **set_params.py:** File to set the simulation parameters.
*  **generate_graph.py:** Functions to create the network topology.

All the codes were implemented in a event-based fashion, where the network state is updated when a spike ocurs.

The two codes below update the whole state variables of the network each time an event occurs. The difference between the two versions is the type of synapse. Both are not plastic but one decays exponential while the other is just a instantaneous jump on the postsynaptic potential.
*  **gl_delta_refrac.py:** Script to run the network with GL neurons with refractory time and connected by delta synapses.
*  **gl_expsyn_refrac.py:** Script to run the network with GL neurons with refractory time and connected by synapses with exponential decay.

The next two codes instead of updating all variables of the system it only updates the neuron who is spiking and the list of postsynaptic neurons connected to this one.
*  **gl_optimized_delta.py:** Script to run the network with GL neurons with refractory time and connected by delta synapses.
*  **gl_optimized_expsyn_.py:** Script to run the network with GL neurons with refractory time and connected by synapses with exponential decay.

The codes with synaptic delay were not added here yet.
