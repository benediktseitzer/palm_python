# -*- coding: utf-8 -*-
################
""" 
author: benedikt.seitzer
name: palm_py.plot
purpose: set global variables and defaults
"""
################

import numpy as np
import pandas as pd

import palm_py as papy

class globals():
    """
    This is the globals class. Globals objects are initialized here.
    It is possible to do changes in values of the global variable.
    """
    def __init__(self, calc_kai_sim=None, testing=None):
        """
        init globals-object and set 
        """
        super().__init__()
        # program steering
        if calc_kai_sim == None:
            calc_kai_sim = False 
            print('testi')
        self.calc_kai_sim = calc_kai_sim

        if testing == None:
            testing = False
        self.testing = testing
        print('test')
        # physical parameters
        self.z0 = None # roughness length
        self.d0 = None # displacement height
        self.ka = None # von Karman constant
        self.alpha = None # fitting parameter power law

        # environmental parameters
        self.run_name = None
        self.run_number = None
