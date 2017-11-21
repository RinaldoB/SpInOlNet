from brian import Connection, NeuronGroup, PoissonGroup, SpikeMonitor, Network, MultiStateMonitor, define_default_clock, run
from brian.stdunits import ms
import numpy as np

def runsim(monitors, neuron_model, orn_rates,
           wORNPN, wORNLN, wLNPN, wPNKC,
           N_glu, ORNperGlu, PNperGlu, LNperGlu,
           N_KC, PNperKC,
           V0min, V0max,
           prerun, simtime, dt, beeid=0,
           inh_delay=0*ms, inh_struct=None, recvars=None, rv_timestep=500, mhook=None, report=False):
    '''
    ## ToDo documentation ##
    Connect ORNs to PNs such that ORNperGlu ORNs representing input to one Glu connects to 1 PN
    repeat for every Glu, using connect_full. Connects ORNs to LNs in the same way.
    '''
    np.random.seed() #needed for numpy/brian when runing parallel sims
    define_default_clock(dt=dt)    
    

    #########################     NEURONGROUPS     #########################

    NG = dict()

    # ORN Input
    NG['ORN'] = PoissonGroup(ORNperGlu*N_glu, rates=orn_rates)
    NG['PN'] =  NeuronGroup(N_glu*PNperGlu, **neuron_model)
    NG['LN'] =  NeuronGroup(N_glu*LNperGlu, **neuron_model)
    if 'KC' in monitors: NG['KC'] = NeuronGroup(N_KC, **neuron_model)

    #########################     CONNECTIONS       #########################
    c = dict()
    
    ### ORN-PN ###
    c['ORNPN'] = Connection(NG['ORN'],NG['PN'],'ge')
    for i in np.arange(N_glu): c['ORNPN'].connect_full(NG['ORN'].subgroup(ORNperGlu),NG['PN'][i],weight=wORNPN)

    ### ORN-LN ###
    c['ORNLN'] = Connection(NG['ORN'],NG['LN'],'ge')
    for i in np.arange(N_glu):
        c['ORNLN'].connect_full(NG['ORN'][ i*ORNperGlu : (i+1)*ORNperGlu ], NG['LN'][i], weight = wORNLN)

    ### LN-PN ###
    c['LNPN'] = Connection(NG['LN'],NG['PN'],'gi',weight=wLNPN, delay=inh_delay)
    if inh_struct: c['LNPN'].connect(NG['LN'],NG['PN'],inh_struct)
    
    ## PN-KC ##
    if 'KC' in monitors:
        c['KC'] = Connection(NG['PN'],NG['KC'],'ge')
        c['KC'].connect_random(NG['PN'],NG['KC'],p=PNperKC/float(N_glu*PNperGlu),weight=wPNKC,seed=beeid)
        # the total number of possible synapses is N_pre*N_post
        # when sparseness is 0.05 then N_syn = N_pre*N_post*0.05 (on average)
        # every postsynaptic neuron will receive N_syn/N_post synaptic inputs _on average_
        # and every presynaptic input will send out N_syn/N_pre _on average_
        # the distribution of synapses is given by the biominal distribution?
    
    #########################     INITIAL VALUES     #########################
    NG['PN'].vm=np.random.uniform(V0min,V0max,size=len(NG['PN']))
    NG['LN'].vm=np.random.uniform(V0min,V0max,size=len(NG['LN']))
    if 'KC' in monitors: NG['KC'].vm=np.random.uniform(V0min,V0max,size=len(NG['KC']))
    
    net = Network(NG.values(), c.values())
    
    #### Insert Hook ###
    if mhook:
        globals().update(mhook(net, NG, c))
    ##########################################################################
    
    #########################         PRE-RUN        #########################    
    net.run(prerun)
    #########################     MONITORS           #########################
    spmons = [SpikeMonitor(NG[mon], record=True) for mon in monitors]
    net.add(spmons)
    
    if recvars is not None:
        mons = [MultiStateMonitor(NG[mon], vars=recvars, record=True, timestep=rv_timestep) for mon in monitors]
        net.add(mons)
    else:
        mons = None
    #########################           RUN          #########################
    net = run(simtime, report=report)
    

    out_spikes = dict( (monitors[i],np.array(sm.spikes)) for i,sm in enumerate(spmons) )
    
    if mons is not None:
        out_mons = dict( (mon,dict((var,statemon.values) for var,statemon in m.iteritems())) for mon,m in zip(monitors,mons))
    else:
        out_mons = None

    #subtract the prerun from spike times, if there are any
    for spikes in out_spikes.itervalues():
        if len(spikes) != 0:
            spikes[:,1] -= prerun
    
    return out_spikes, out_mons
