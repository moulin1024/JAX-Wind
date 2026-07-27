#!/usr/bin/env python
'''
Created on 18.04.2018

@author: trevaz (tristan.revaz@epfl.ch)

---------------------------------------------------------------------------------
app: clean
---------------------------------------------------------------------------------
'''

#################################################################################
# IMPORTS
#################################################################################
import os
from fctlib import get_case_path

#################################################################################
# CONSTANTS
#################################################################################


#################################################################################
# MAIN FUNCTION
#################################################################################
def clean(PATH, case_name):
    '''
    DEF:    clean case.
    INPUT:  - case_name: name of the case, type=string
    OUTPUT: - ()
    '''
    case_path = get_case_path(PATH, case_name)
    os.system('rm -r ' + os.path.join(case_path, 'src ')  + \
    os.path.join(case_path, 'output ') + \
    os.path.join(case_path, 'init_data ') + \
    os.path.join(case_path, 'inflow_data ') + \
    os.path.join(case_path, 'slurm* ') + \
    os.path.join(case_path, 'log'))
