'''
Created on 18.04.2018

@author: trevaz

---------------------------------------------------------------------------------
app: pre-process
---------------------------------------------------------------------------------
'''

#################################################################################
# IMPORTS
#################################################################################
import os
import fctlib
import numpy as np
from string import Template
# from scipy.interpolate import RegularGridInterpolator
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


#################################################################################
# CONSTANTS
#################################################################################


#################################################################################
# MAIN FUNCTION
#################################################################################
def pre(PATH, case_name):
    '''
    DEF:    pre-processing for wireles.
    INPUT:  - case_name
    OUTPUT: - ()
    '''
    case_path = fctlib.get_case_path(PATH, case_name)

    ############################################################################
    # COPY SRC FOLDER
    os.system('cp -r ' + PATH['src'] + ' ' + case_path)

    ############################################################################
    # EXTRACT CONFIG
    print('extract config...')
    config = fctlib.get_config(case_path)
    space = fctlib.get_space(config)

    ############################################################################
    # WRITE CONFIG
    print('write dimen...')
    config_path_tmp = os.path.join(PATH['src'],'dimen.cuf')
    config_path = os.path.join(case_path, 'src','dimen.cuf')
    config_param = config.copy()
    substitute_keys(config_param, config_path_tmp, config_path)

    ############################################################################
    # WRITE BASH
    print('write jobsub...')
    sbatch_path_tmp = os.path.join(PATH['src'], 'jobsub')
    sbatch_path = os.path.join(case_path, 'src', case_name + '_src')
    sbatch_param = { 'case_name':case_name, 'job_np': config['job_np'], 'job_time':config['job_time']}
    substitute_keys(sbatch_param, sbatch_path_tmp, sbatch_path)

    ############################################################################
    # WRITE INIT
    print('write init...')

    if config['sim_flag'] == 0:

        if config['resub_flag'] == 0:
            u_init, v_init, w_init = compute_vel(config)
            u_init.tofile(os.path.join(case_path, 'src','input','u.bin'))
            v_init.tofile(os.path.join(case_path, 'src','input','v.bin'))
            w_init.tofile(os.path.join(case_path, 'src','input','w.bin'))

    if config['sim_flag'] == 1:

        if config['resub_flag'] == 0:

            interp_path = fctlib.get_case_path(PATH, config['interp_case'])
            u_init = post_interp(interp_path, case_path, config, 'u')
            v_init = post_interp(interp_path, case_path, config, 'v')
            w_init = post_interp(interp_path, case_path, config, 'w')
            u_init.tofile(os.path.join(case_path, 'src','input','u.bin'))
            v_init.tofile(os.path.join(case_path, 'src','input','v.bin'))
            w_init.tofile(os.path.join(case_path, 'src','input','w.bin'))

    if config['sim_flag'] == 2:

        if config['resub_flag'] == 0:

            warmup_path = fctlib.get_case_path(PATH, config['warmup_case'])
            os.makedirs(os.path.join(case_path, 'inflow_data'))
            os.system('cp ' + os.path.join(warmup_path, 'init_data/*') + ' ' + os.path.join(case_path, 'src','input'))
            os.system('cp -r ' + os.path.join(warmup_path, 'init_data') + ' ' + os.path.join(case_path))

    if config['sim_flag'] == 3:

        prec_path = fctlib.get_case_path(PATH, config['prec_case'])
        pwd = os.getcwd()
        if config['resub_flag'] == 0:
            os.system('cp ' + os.path.join(prec_path, 'init_data/*') + ' ' + os.path.join(case_path, 'src', 'input'))
        # os.system('cp -r ' + os.path.join(prec_path, 'inflow_data') + ' ' + os.path.join(case_path))
        os.system('ln -snf ' + os.path.join(pwd, prec_path, 'inflow_data') + ' ' + os.path.join(pwd, case_path, 'inflow_data'))

    if config['sim_flag'] == 4:

        prec_path = fctlib.get_case_path(PATH, config['prec_case'])
        inflow_path = os.path.join(prec_path,'inflow_data_bis')
        pwd = os.getcwd()
        if config['resub_flag'] == 0:
            os.makedirs(os.path.join(case_path, 'inflow_data'))
            os.system('cp ' + os.path.join(prec_path, 'init_data/*') + ' ' + os.path.join(case_path, 'src', 'input'))
        if not os.path.isdir(inflow_path):
            os.mkdir(inflow_path)
            for i in range(config['job_np']):
                post_prec(prec_path, config,'u',i)
                post_prec(prec_path, config,'v',i)
                post_prec(prec_path, config,'w',i)
        os.system('ln -snf ' + os.path.join(pwd, inflow_path, '*') + ' ' + os.path.join(pwd, case_path, 'inflow_data'))


    if config['turb_flag'] == 1:
        os.system('cp ' +os.path.join(case_path, 'input/*.dat')+ ' '+ os.path.join(case_path, 'src', 'input'))
        os.system('cp ' +os.path.join(case_path, 'input/*.csv')+ ' '+ os.path.join(case_path, 'src', 'input'))

    # write zo and zs. Must be here to allow for changes beween prec/main

    zo = compute_zo(config)
    zo.tofile(os.path.join(case_path, 'src','input','zo.bin'))

    print('----------------------------------')
    print(' SPACE PARAMS:')
    print(' nx x ny x nz    : ' + str(config['nx']) + ' x ' + str(config['ny']) + ' x ' + str(config['nz']))
    print(' z_i x l_z x l_r : ' + str(config['z_i']) + ' x ' + str(config['l_z']) + ' x ' + str(config['l_r']))
    print(' lx x ly x lz    : ' + str(config['lx']) + ' x ' + str(config['ly']) + ' x ' + str(config['lz']) )
    print(' dx x dy x dz    : ' + str(config['dx']) + ' x ' + str(config['dy']) + ' x ' + str(config['dz']) )
    print('----------------------------------')
    print(' TIME PARAMS:')
    print(' nsteps : ' + str(config['nsteps']))
    print(' dt     : ' + str(config['dt']))
    print(' t_tot  : ' + str(config['t_tot']))
    print(' dtr    : ' + str(config['dtr']))
    print('----------------------------------')

#################################################################################
# SECONDARY FUNCTIONS
#################################################################################
def compute_vel(config):

    # INIT
    u = np.zeros((config['nx'],config['ny'],config['nz']))
    v = np.zeros((config['nx'],config['ny'],config['nz']))
    w = np.zeros((config['nx'],config['ny'],config['nz']))


    # ADD MEAN
    zc = np.linspace(0, config['l_z'], config['nz']) + 0.5*config['l_z']/(config['nz']-1)
    u = np.tile((config['u_fric']/0.4)*(np.log(zc/config['zo'])),(config['nx'], config['ny'] ,1))

    # ADD RAND
    u[:,:,0:4] = u[:,:,0:4] + 0.1*(np.random.rand(config['nx'],config['ny'],4)-0.5)
    # r2 = np.random.rand(nx,ny,nz)-0.5
    # r3 = np.random.rand(nx,ny,nz)-0.5
    # V2 = 0.*r3
    # W2 = 0.*r3

    # correct for top bc
    u[:,:,-1]=u[:,:,-2]

    u[:,:,int(config['bl_height']//config['dz']):]=u[0,0,int(config['bl_height']//config['dz'])]

    # RESHAPE
    if config['double_flag'] ==0:
        print('u,v,w: Single precision!')
        u = np.ravel(np.float32(u), order='F')
        v = np.ravel(np.float32(v), order='F')
        w = np.ravel(np.float32(w), order='F')
    else:
        print('u,v,w: Double precision!')
        u = np.ravel(np.float64(u), order='F')
        v = np.ravel(np.float64(v), order='F')
        w = np.ravel(np.float64(w), order='F')

    return u, v, w

def compute_zo(config):

    # INIT
    if config['double_flag'] ==0:
        print('zo: Single precision!')
        zo = np.ones((config['nx'],config['ny']))*config['zo']
        zo = np.ravel(np.float32(zo), order='F')
    else:
        print('zo: Double precision!')
        zo = np.ones((config['nx'],config['ny']))*config['zo']
        zo = np.ravel(np.float64(zo), order='F')
    
    

    return zo

def substitute_keys(mydict, mytemplate_path, myfile_path):
    myfile_tmp = Template(fctlib.read_file(mytemplate_path))
    for mykey in (mydict):
        myfile = myfile_tmp.safe_substitute({mykey: mydict[mykey]})
        myfile_tmp = Template(myfile)
    fctlib.write_file(myfile, os.path.join(myfile_path))

# def post_interp(interp_path, case_path, config, var_name):
#     interp_config = fctlib.get_config(interp_path)
#     interp_u = fctlib.load_3d(var_name, interp_config['nx'],  interp_config['ny'],  interp_config['nz'],  os.path.join(interp_path, 'init_data'))
#     X, Y, Z = np.meshgrid(np.linspace(0,config['lx'],config['nx']), np.linspace(0,config['ly'],config['ny']), np.linspace(0,config['lz'],config['nz']), indexing='ij')
#     F = RegularGridInterpolator((np.linspace(0,interp_config['lx'],interp_config['nx']), np.linspace(0,interp_config['ly'],interp_config['ny']), np.linspace(0,interp_config['lz'],interp_config['nz'])), interp_u, method='linear')
#     warmup_u = F((X, Y, Z))
#     warmup_u = np.ravel(warmup_u, order='F')

#     return warmup_u


def post_prec(prec_path, config,var_name,i):
    print(' * proc_id: ' + str(i))
    print(' * var_id : ' + var_name)
    print('   read...')
    inflow_u = fctlib.load_4d('p'+ str(i).zfill(3) + '_inflow_' + var_name, config['nsteps']//config['inflow_count'], config['inflow_nx'],  config['ny'],  config['nzb'],config['double_flag'],  os.path.join(prec_path, 'inflow_data'))
    # os.system('rm ' +os.path.join(prec_data_path,'p'+ str(i).zfill(2) + '_inflow_' + var_name + '.bin'))
    print('   compute...')
    ta_u= np.mean(inflow_u, axis = 0)
    tvar_u= np.mean((inflow_u-ta_u)**2, axis = 0)
    xya_ta_u= np.mean(ta_u, axis = (0,1))
    xya_tvar_u= np.mean(tvar_u, axis = (0,1))
    scale = (xya_tvar_u/tvar_u)**0.5
    if var_name == 'w' and i==0:
        scale[:,:,0] = 1
    if var_name == 'w' and i==config['job_np']-1:
        scale[:,:,-1] = 1
    inflow_u = (inflow_u - ta_u)*scale + xya_ta_u
    inflow_u = np.ravel(np.transpose(inflow_u, (1,2,3,0)), order='F')
    print('   write...')
    inflow_u.tofile(os.path.join(prec_path,'inflow_data_bis','p'+ str(i).zfill(3) + '_inflow_' + var_name + '.bin'))
