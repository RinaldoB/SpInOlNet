#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:45:26 2017

@author: rinaldo
"""

from brian import MembraneEquation,  Current, IonicCurrent
import numpy as np

#from brian.library.IF import *
from brian.library.ionic_currents import leak_current
from brian.library.synapses import exp_conductance

def model_aIF(C,gL,EL,a,b,tauw,Ee,tau_syn_e,Ei,tau_syn_i,VT,Vr,tau_ref):
    '''
    returns a neuron model that can be passed to NeuronGroup
    '''
    ## ------- Model ------- ##
    neuron_model = dict()
    neuron_model['model'] = MembraneEquation(C=C) + leak_current(gl=gL,El=EL) 
    neuron_model['model'] += IonicCurrent('dw/dt=(a*(vm-EL)-w)/tauw:amp', a=a, EL=EL, tauw=tauw)    
    neuron_model['model'] += Current('I0: amp')
    neuron_model['model'] += exp_conductance('ge', Ee, tau_syn_e)
    neuron_model['model'] += exp_conductance('gi', Ei, tau_syn_i)

    neuron_model['threshold'] = VT
    ### WARNING ### Multi Variable Reset is broken
    ### see https://groups.google.com/forum/#!topic/briansupport/IcpATns3X38
    neuron_model['reset']   = '''vm  = Vr
                                 w  += b'''
    neuron_model['refractory'] = tau_ref
    
    return neuron_model

def model_saIF(C, gL, EL, a, b, tauw, Ee, tau_syn_e, Ei, tau_syn_i, VT, Vr, tau_ref, D):
    '''
    returns a neuron model that can be passed to NeuronGroup
    '''
    
    sigma = np.sqrt(2*D)*b
    
    ## ------- Model ------- ##
    neuron_model = dict()
    neuron_model['model'] = MembraneEquation(C=C) + leak_current(gl=gL,El=EL) 
    neuron_model['model'] += IonicCurrent('dw/dt=(a*(vm-EL)-w)/tauw + sigma*xi/tauw**.5 :amp', a=a, EL=EL, tauw=tauw, sigma=sigma)    
    neuron_model['model'] += Current('I0: amp')
    neuron_model['model'] += exp_conductance('ge', Ee, tau_syn_e)
    neuron_model['model'] += exp_conductance('gi', Ei, tau_syn_i)

    neuron_model['threshold'] = VT
    ### WARNING ### Multi Variable Reset is broken
    ### see https://groups.google.com/forum/#!topic/briansupport/IcpATns3X38
    neuron_model['reset']   = '''vm  = Vr
                                 w  += b'''
    neuron_model['refractory'] = tau_ref
    
    return neuron_model


def model_IF(C, gL, EL, Ee, tau_syn_e, Ei, tau_syn_i, VT, Vr, tau_ref):
    '''
    returns a neuron model that can be passed to NeuronGroup
    '''
    ## ------- Model ------- ##
    neuron_model = dict()
    neuron_model['model'] = MembraneEquation(C=C) + leak_current(gl=gL,El=EL) 
    #neuron_model['model'] += IonicCurrent('dw/dt=(a*(vm-EL)-w)/tauw:amp', a=a, EL=EL, tauw=tauw)    
    neuron_model['model'] += Current('I0: amp')
    neuron_model['model'] += exp_conductance('ge', Ee, tau_syn_e)
    neuron_model['model'] += exp_conductance('gi', Ei, tau_syn_i)

    neuron_model['threshold'] = VT
    neuron_model['reset']     = Vr
    neuron_model['refractory'] = tau_ref
    ### WARNING ### Multi Variable Reset is broken
    ### see https://groups.google.com/forum/#!topic/briansupport/IcpATns3X38
    #neuron_model['reset']   = '''vm  = Vr
    #                             w  += b'''    

    return neuron_model