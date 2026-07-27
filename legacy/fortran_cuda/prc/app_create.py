#!/usr/bin/env python
'''
Created on 18.04.2018

@author: trevaz (tristan.revaz@epfl.ch)

---------------------------------------------------------------------------------
app: create
---------------------------------------------------------------------------------
'''

#################################################################################
# IMPORTS
#################################################################################
import os

#################################################################################
# CONSTANTS
#################################################################################


#################################################################################
# MAIN FUNCTION
#################################################################################
def create(PATH, case_name):
    '''
    DEF:    create case.
    INPUT:  - case_name: name of the case, type=string
    OUTPUT: - ()
    '''

    case_path = os.path.join(PATH['job'], case_name)
    case_input_path = os.path.join(case_path, 'input')

    if not os.path.isdir(case_path):
        os.makedirs(case_input_path)
        os.system('cp ' + os.path.join(PATH['prc'], 'config') + ' ' + case_input_path)
        # os.system('tail -f ' + os.path.join(case_input_path, 'config'))

    else :
        print(' --> case already exist')
