'''
Created on 18.04.2018

@author: trevaz

--------------------------------------------------------------------------------
app: post-process
--------------------------------------------------------------------------------
'''

################################################################################
# IMPORT
################################################################################
import os, sys
import fctlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

################################################################################
# CONSTANTS
################################################################################

# REF
turb_i = 65-1
turb_j = 33-1

################################################################################
# MAIN FONCTION
################################################################################
def post(PATH, case_name):
    '''
    DEF:    post-processing for wireles.
    INPUT:  - case_name
    OUTPUT: - ()
    '''
    case_path = fctlib.get_case_path(PATH, case_name)

    ############################################################################
    # INIT
    out_path = os.path.join(case_path, 'output')
    fctlib.test_and_mkdir(out_path)
    src_out_path = os.path.join(PATH['job'], case_name, 'src', 'output')
    src_inp_path = os.path.join(PATH['job'], case_name, 'src', 'input')

    ############################################################################
    # CONFIG
    print('extract config...')
    config = fctlib.get_config(case_path)
    space = get_space(config)
    time = get_time(config)

    ############################################################################
    # COMPUTE RESULTS
    print('compute results...')
    if config['log_flag'] > 0:
        log = get_log(src_out_path, config)
    if config['ta_flag'] > 0:
        result_3d = get_result_3d(src_inp_path, src_out_path, config)
        result_pr = get_result_pr(result_3d, config)
    if config['ts_flag'] > 0:
        result_4d = get_result_4d(src_out_path, config)
    if config['turb_flag'] > 0:
        turb = get_turb(src_out_path, config)

    ############################################################################
    # NORMALIZATION

    turb_k = int(config['turb_z']/config['dz'])
    result_3d['u_avgInflow'] = np.mean(result_3d['u_avg_c'][6-1:10-1,:,:],axis=(0,1))
    result_3d['u_stdInflow'] = np.mean(result_3d['u_std_c'][6-1:10-1,:,:],axis=(0,1))
    U = result_3d['u_avgInflow'][turb_k]
    D = 2*config['turb_r']

    print( '----------------------------------' )
    print( 'turb_z     : ' + str(config['turb_z']))
    print( 'turb_k     : ' + str(turb_k))
    print( 'z_c[turb_k]: ' + str(space['z_c'][turb_k]) )
    print( '----------------------------------' )
    print( 'U: ' + str(U) )
    print( 'D: ' + str(D) )
    print('----------------------------------')

    space['x'] = (space['x']-turb_i*config['dx'])/D
    space['y'] = (space['y']-turb_j*config['dy'])/D
    space['x_'] = (space['x_']-turb_i*config['dx'])/D
    space['y_'] = (space['y_']-turb_j*config['dy'])/D
    space['z_c'] = space['z_c']/D
    space['z_n'] = space['z_n']/D

    ############################################################################
    # PLOT RESULTS
    print('plot results...')

    if config['log_flag'] > 0:
        plot_log(time, log, config, out_path)

    if config['turb_flag'] > 0:
        plot_turb(time, turb, config, out_path)

    if config['ta_flag'] > 0:

        ########################################################################
        # pr global
        ########################################################################

        plot_pr_uvw(space, result_pr, config, out_path)
        plot_pr_log(space, result_pr, config, out_path)
        plot_pr_phi(space, result_pr, config, out_path)
        plot_pr_st(space, result_pr, config, out_path)

        ########################################################################
        # pr local
        ########################################################################
        plot_pr(space['z_c'], result_3d['u_avg_c'][turb_i+2*3 ,turb_j,:], 'z', 'u_avg2', out_path)
        plot_pr(space['z_c'], result_3d['u_avg_c'][turb_i+4*3 ,turb_j,:], 'z', 'u_avg4', out_path)
        plot_pr(space['z_c'], result_3d['u_avg_c'][turb_i+6*3 ,turb_j,:], 'z', 'u_avg6', out_path)
        plot_pr(space['z_c'], result_3d['u_avg_c'][turb_i+8*3 ,turb_j,:], 'z', 'u_avg8', out_path)

        plot_pr(space['y'], result_3d['u_avg_c'][turb_i+2*3 ,:,turb_k], 'y', 'u_avg2', out_path)
        plot_pr(space['y'], result_3d['u_avg_c'][turb_i+4*3 ,:,turb_k], 'y', 'u_avg4', out_path)
        plot_pr(space['y'], result_3d['u_avg_c'][turb_i+6*3 ,:,turb_k], 'y', 'u_avg6', out_path)
        plot_pr(space['y'], result_3d['u_avg_c'][turb_i+8*3 ,:,turb_k], 'y', 'u_avg8', out_path)

        plot_pr(space['z_c'], result_3d['u_std_c'][turb_i+2*3 ,turb_j,:], 'z', 'u_std2', out_path)
        plot_pr(space['z_c'], result_3d['u_std_c'][turb_i+4*3 ,turb_j,:], 'z', 'u_std4', out_path)
        plot_pr(space['z_c'], result_3d['u_std_c'][turb_i+6*3 ,turb_j,:], 'z', 'u_std6', out_path)
        plot_pr(space['z_c'], result_3d['u_std_c'][turb_i+8*3 ,turb_j,:], 'z', 'u_std8', out_path)

        plot_pr(space['y'], result_3d['u_std_c'][turb_i+2*3 ,:,turb_k], 'y', 'u_std2', out_path)
        plot_pr(space['y'], result_3d['u_std_c'][turb_i+4*3 ,:,turb_k], 'y', 'u_std4', out_path)
        plot_pr(space['y'], result_3d['u_std_c'][turb_i+6*3 ,:,turb_k], 'y', 'u_std6', out_path)
        plot_pr(space['y'], result_3d['u_std_c'][turb_i+8*3 ,:,turb_k], 'y', 'u_std8', out_path)

        plot_pr(space['z_c'], result_3d['v_avg_c'][turb_i+2*3 ,turb_j,:], 'z', 'v_avg2', out_path)
        plot_pr(space['z_c'], result_3d['v_avg_c'][turb_i+4*3 ,turb_j,:], 'z', 'v_avg4', out_path)
        plot_pr(space['z_c'], result_3d['v_avg_c'][turb_i+6*3 ,turb_j,:], 'z', 'v_avg6', out_path)
        plot_pr(space['z_c'], result_3d['v_avg_c'][turb_i+8*3 ,turb_j,:], 'z', 'v_avg8', out_path)

        plot_pr(space['y'], result_3d['w_avg_c'][turb_i+2*3 ,:,turb_k], 'y', 'w_avg2', out_path)
        plot_pr(space['y'], result_3d['w_avg_c'][turb_i+4*3 ,:,turb_k], 'y', 'w_avg4', out_path)
        plot_pr(space['y'], result_3d['w_avg_c'][turb_i+6*3 ,:,turb_k], 'y', 'w_avg6', out_path)
        plot_pr(space['y'], result_3d['w_avg_c'][turb_i+8*3 ,:,turb_k], 'y', 'w_avg8', out_path)

        ########################################################################
        # sl
        ########################################################################

        plot_sl(space['x_'], space['y_'], result_3d['u_avg_c'][:,:,turb_k], 'x', 'y', 'u_avg', 1, out_path)
        plot_sl(space['x_'], space['z_n'], result_3d['u_avg_c'][:,turb_j,:], 'x', 'z', 'u_avg', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['u_avg_c'][turb_i,:,:], 'y', 'z', 'u_avg', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['u_avg_c'][turb_i+2*3  ,:,:], 'y', 'z', 'u_avg2', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['u_avg_c'][turb_i+4*3 ,:,:], 'y', 'z', 'u_avg4', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['u_avg_c'][turb_i+6*3 ,:,:], 'y', 'z', 'u_avg6', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['u_avg_c'][turb_i+8*3 ,:,:], 'y', 'z', 'u_avg8', 1, out_path)

        plot_sl(space['x_'], space['y_'], result_3d['v_avg_c'][:,:,turb_k], 'x', 'y', 'v_avg', 1, out_path)
        plot_sl(space['x_'], space['z_n'], result_3d['v_avg_c'][:,turb_j,:], 'x', 'z', 'v_avg', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['v_avg_c'][turb_i,:,:], 'y', 'z', 'v_avg', 1, out_path)

        plot_sl(space['x_'], space['y_'], result_3d['w_avg_c'][:,:,turb_k], 'x', 'y', 'w_avg', 1, out_path)
        plot_sl(space['x_'], space['z_n'], result_3d['w_avg_c'][:,turb_j,:], 'x', 'z', 'w_avg', 1, out_path)
        plot_sl(space['y_'], space['z_n'], result_3d['w_avg_c'][turb_i,:,:], 'y', 'z', 'w_avg', 1, out_path)

        ########################################################################
        # save pr for vali
        ########################################################################
        out_data_path = os.path.join(out_path,'data')
        fctlib.test_and_mkdir(out_data_path)

        save_pr(space['z_c'], result_3d['u_avgInflow'], 'z_u_avgInflow', out_data_path)
        save_pr(space['z_c'], result_3d['u_stdInflow'], 'z_u_stdInflow', out_data_path)

        save_pr(space['z_c'], result_3d['u_avg_c'][turb_i+2*3 ,turb_j,:], 'z_u_avg2', out_data_path)
        save_pr(space['z_c'], result_3d['u_avg_c'][turb_i+4*3 ,turb_j,:], 'z_u_avg4', out_data_path)
        save_pr(space['z_c'], result_3d['u_avg_c'][turb_i+6*3 ,turb_j,:], 'z_u_avg6', out_data_path)
        save_pr(space['z_c'], result_3d['u_avg_c'][turb_i+8*3 ,turb_j,:], 'z_u_avg8', out_data_path)

        save_pr(space['y'], result_3d['u_avg_c'][turb_i+2*3 ,:,turb_k], 'y_u_avg2', out_data_path)
        save_pr(space['y'], result_3d['u_avg_c'][turb_i+4*3 ,:,turb_k], 'y_u_avg4', out_data_path)
        save_pr(space['y'], result_3d['u_avg_c'][turb_i+6*3 ,:,turb_k], 'y_u_avg6', out_data_path)
        save_pr(space['y'], result_3d['u_avg_c'][turb_i+8*3 ,:,turb_k], 'y_u_avg8', out_data_path)

        save_pr(space['z_c'], result_3d['u_std_c'][turb_i+2*3 ,turb_j,:], 'z_u_std2', out_data_path)
        save_pr(space['z_c'], result_3d['u_std_c'][turb_i+4*3 ,turb_j,:], 'z_u_std4', out_data_path)
        save_pr(space['z_c'], result_3d['u_std_c'][turb_i+6*3 ,turb_j,:], 'z_u_std6', out_data_path)
        save_pr(space['z_c'], result_3d['u_std_c'][turb_i+8*3 ,turb_j,:], 'z_u_std8', out_data_path)

        save_pr(space['y'], result_3d['u_std_c'][turb_i+2*3 ,:,turb_k], 'y_u_std2', out_data_path)
        save_pr(space['y'], result_3d['u_std_c'][turb_i+4*3 ,:,turb_k], 'y_u_std4', out_data_path)
        save_pr(space['y'], result_3d['u_std_c'][turb_i+6*3 ,:,turb_k], 'y_u_std6', out_data_path)
        save_pr(space['y'], result_3d['u_std_c'][turb_i+8*3 ,:,turb_k], 'y_u_std8', out_data_path)

        save_pr(space['z_c'], result_3d['v_avg_c'][turb_i+2*3 ,turb_j,:], 'z_v_avg2', out_data_path)
        save_pr(space['z_c'], result_3d['v_avg_c'][turb_i+4*3 ,turb_j,:], 'z_v_avg4', out_data_path)
        save_pr(space['z_c'], result_3d['v_avg_c'][turb_i+6*3 ,turb_j,:], 'z_v_avg6', out_data_path)
        save_pr(space['z_c'], result_3d['v_avg_c'][turb_i+8*3 ,turb_j,:], 'z_v_avg8', out_data_path)

        save_pr(space['y'], result_3d['w_avg_c'][turb_i+2*3 ,:,turb_k], 'y_w_avg2', out_data_path)
        save_pr(space['y'], result_3d['w_avg_c'][turb_i+4*3 ,:,turb_k], 'y_w_avg4', out_data_path)
        save_pr(space['y'], result_3d['w_avg_c'][turb_i+6*3 ,:,turb_k], 'y_w_avg6', out_data_path)
        save_pr(space['y'], result_3d['w_avg_c'][turb_i+8*3 ,:,turb_k], 'y_w_avg8', out_data_path)


        ########################################################################
        # fluct
        ########################################################################

        # plot_sl(space['x_'], space['y_'], result_3d['u_inst_c'][:,:,turb_k]-result_3d['u_avg_c'][:,:,turb_k], 'x', 'y', 'u_fluct', 1, out_path)
        # plot_sl(space['x_'], space['z_n'], result_3d['u_inst_c'][:,turb_j,:]-result_3d['u_avg_c'][:,turb_j,:], 'x', 'z', 'u_fluct', 1, out_path)
        #
        # plot_sl(space['x_'], space['y_'], result_3d['v_inst_c'][:,:,turb_k]-result_3d['v_avg_c'][:,:,turb_k], 'x', 'y', 'v_fluct', 1, out_path)
        # plot_sl(space['x_'], space['z_n'], result_3d['v_inst_c'][:,turb_j,:]-result_3d['v_avg_c'][:,turb_j,:], 'x', 'z', 'v_fluct', 1, out_path)
        #
        # plot_sl(space['x_'], space['y_'], result_3d['w_inst_n'][:,:,turb_k]-result_3d['w_avg_n'][:,:,turb_k], 'x', 'y', 'w_fluct', 1, out_path)
        # plot_sl(space['x_'], space['z_n'], result_3d['w_inst_n'][:,turb_j,:]-result_3d['w_avg_n'][:,turb_j,:], 'x', 'z', 'w_fluct', 1, out_path)

    # if config['ts_flag'] > 0:
    #
    #     plot_sl_anim(space['x_'], space['z_n'], result_4d['u_inst_c'][:,:,turb_j,:], 'x', 'z', 'u_inst', 1, out_path)
    #     plot_sl_anim(space['x_'], space['y_'], result_4d['u_inst_c'][:,:,:,turb_k], 'x', 'y', 'u_inst', 1, out_path)
    #
    #     plot_sl_anim(space['x_'], space['z_n'], result_4d['v_inst_c'][:,:,turb_j,:], 'x', 'z', 'v_inst', 1, out_path)
    #     plot_sl_anim(space['x_'], space['y_'], result_4d['v_inst_c'][:,:,:,turb_k], 'x', 'y', 'v_inst', 1, out_path)
    #
    #     plot_sl_anim(space['x_'], space['z_n'], result_4d['w_inst_c'][:,:,turb_j,:]-result_3d['w_avg_c'][:,turb_j,:], 'x', 'z', 'w_inst', 1, out_path)
    #     plot_sl_anim(space['x_'], space['y_'], result_4d['w_inst_c'][:,:,:,turb_k]-result_3d['w_avg_c'][:,:,turb_k], 'x', 'y', 'w_inst', 1, out_path)

#################################################################################
# PROCESS FUNCTIONS
#################################################################################

def get_space(config):

    space = {}
    space['x'] = np.arange(0, config['lx'], config['dx'])
    space['y'] = np.arange(0, config['ly'], config['dy'])
    space['z_n'] = np.arange(0, config['lz']+config['dz'], config['dz'])
    space['z_c'] = node2center_1d(space['z_n'])
    space['x_'] = np.arange(0, config['lx']+config['dx'], config['dx']) - 0.5*config['dx']
    space['y_'] = np.arange(0, config['ly']+config['dy'], config['dy']) - 0.5*config['dy']

    return space

def get_time(config):

    time = {}
    time['t'] = config['dtr']*np.arange(0,config['nsteps'])
    time['t_ta'] = config['p_count']*config['dtr']*np.arange(config['ta_tstart'],config['ta_tstart']+config['ta_ns']+1)
    time['t_ts'] = config['c_count']*config['dtr']*np.arange(config['ts_tstart'],config['ts_tstart']+config['ts_ns']+1)

    return time

def get_log(src_out_path, config):

    log = {}

    log['ustar'] = fctlib.load_1d('log_ustar', config['nsteps'], src_out_path)
    log['umax']  = fctlib.load_1d('log_umax', config['nsteps'], src_out_path)

    return log

def get_turb(src_out_path, config):

    turb = {}

    turb['thrust'] = fctlib.load_1d('turb_thrust', config['nsteps'], src_out_path)
    turb['power']  = fctlib.load_1d('turb_power', config['nsteps'], src_out_path)

    return turb

def get_result_3d(src_inp_path, src_out_path, config):

    result_3d = {}

    # avg

    ta_u = fctlib.load_4d('ta_u', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)
    ta_v  = fctlib.load_4d('ta_v', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)
    ta_w = fctlib.load_4d('ta_w', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)

    ta_uu  = fctlib.load_4d('ta_u2', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)
    ta_vv  = fctlib.load_4d('ta_v2', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)
    ta_ww = fctlib.load_4d('ta_w2', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)
    ta_uw = fctlib.load_4d('ta_uw', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)

    ta_txz = fctlib.load_4d('ta_txz', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)

    ta_dudz = fctlib.load_4d('ta_dudz', config['ta_ns'], config['nx'],  config['ny'],  config['nz'], src_out_path)

    result_3d['u_avg_c'] = ta_u[-1,:,:,:-1]
    result_3d['u_avg_n'] = center2node_3d(ta_u[-1,:,:,:])
    result_3d['v_avg_c'] = ta_v[-1,:,:,:-1]
    result_3d['v_avg_n'] = center2node_3d(ta_v[-1,:,:,:])
    result_3d['w_avg_c'] = node2center_3d(ta_w[-1,:,:,:])
    result_3d['w_avg_n'] = ta_w[-1,:,:,:]

    result_3d['u_std_c'] = np.sqrt(ta_uu[-1,:,:,:-1]-ta_u[-1,:,:,:-1]*ta_u[-1,:,:,:-1])
    result_3d['u_std_n'] = center2node_3d(np.sqrt(ta_uu[-1,:,:,:]-ta_u[-1,:,:,:]*ta_u[-1,:,:,:]))
    result_3d['v_std_c'] = np.sqrt(ta_vv[-1,:,:,:-1]-ta_v[-1,:,:,:-1]*ta_v[-1,:,:,:-1])
    result_3d['v_std_n'] = center2node_3d(np.sqrt(ta_vv[-1,:,:,:]-ta_v[-1,:,:,:]*ta_v[-1,:,:,:]))
    result_3d['w_std_c'] = node2center_3d(np.sqrt(ta_ww[-1,:,:,:]-ta_w[-1,:,:,:]*ta_w[-1,:,:,:]))
    result_3d['w_std_n'] = np.sqrt(ta_ww[-1,:,:,:]-ta_w[-1,:,:,:]*ta_w[-1,:,:,:])

    result_3d['uw_cov_c'] = node2center_3d(ta_uw[-1,:,:,:]-result_3d['u_avg_n']*result_3d['w_avg_n'])
    result_3d['uw_cov_n'] = ta_uw[-1,:,:,:]-result_3d['u_avg_n']*result_3d['w_avg_n']

    result_3d['txz_avg_c'] = node2center_3d(ta_txz[-1,:,:,:])
    result_3d['txz_avg_n'] = ta_txz[-1,:,:,:]

    result_3d['dudz_avg_c'] = node2center_3d(ta_dudz[-1,:,:,:])
    result_3d['dudz_avg_n'] = ta_dudz[-1,:,:,:]

    # INST

    u  = fctlib.load_3d('u', config['nx'],  config['ny'],  config['nz'], src_inp_path)
    v  = fctlib.load_3d('v', config['nx'],  config['ny'],  config['nz'], src_inp_path)
    w = fctlib.load_3d('w', config['nx'],  config['ny'],  config['nz'], src_inp_path)

    result_3d['u_inst_c'] = u[:,:,:-1]
    result_3d['u_inst_n'] = center2node_3d(u)

    result_3d['v_inst_c'] = v[:,:,:-1]
    result_3d['v_inst_n'] = center2node_3d(v)

    result_3d['w_inst_c'] = node2center_3d(w)
    result_3d['w_inst_n'] = w


    return result_3d

def get_result_4d(src_out_path, config):

    result_4d= {}
    result_4d['u_inst_c'] = fctlib.load_4d('ts_u', config['ts_ns'], config['nx'], config['ny'], config['nz'], src_out_path)[:,:,:,:-1]
    result_4d['v_inst_c'] = fctlib.load_4d('ts_v', config['ts_ns'], config['nx'], config['ny'], config['nz'], src_out_path)[:,:,:,:-1]
    result_4d['w_inst_c'] = node2center_4d(fctlib.load_4d('ts_w', config['ts_ns'], config['nx'], config['ny'], config['nz'], src_out_path))

    return result_4d

def get_result_pr(result_3d, config):

    result_pr = {}
    for key in ('u_avg_c', 'v_avg_c', 'w_avg_n', 'u_std_c', 'v_std_c','w_std_n', 'uw_cov_n', 'txz_avg_n','dudz_avg_n'):
        result_pr[key] = np.mean(result_3d[key], axis=(0,1))

    for key in ('u_inst_c', 'v_inst_c', 'w_inst_n'):
        result_pr[key] = result_3d[key][config['nx']//2, config['ny']//2,:]

    return result_pr

def node2center_4d(var_n):
    var_c = 0.5*(var_n[:,:,:,:-1]+var_n[:,:,:,1:])
    return var_c

def node2center_3d(var_n):
    var_c = 0.5*(var_n[:,:,:-1]+var_n[:,:,1:])
    return var_c

def node2center_1d(var_n):
    var_c = 0.5*(var_n[:-1]+var_n[1:])
    return var_c

def center2node_4d(var_c):
    var_n = np.copy(var_c)
    var_n[:,:,:,0] = 0.0
    var_n[:,:,:,1:] = 0.5*(var_c[:,:,:,:-1]+var_c[:,:,:,1:])
    return var_n

def center2node_3d(var_c):
    var_n = np.copy(var_c)
    var_n[:,:,0] = 0.0
    var_n[:,:,1:] = 0.5*(var_c[:,:,:-1]+var_c[:,:,1:])
    return var_n

####################
# PLOT GENERAL
####################

def plot_option():
    '''
    '''

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('axes', labelsize=14)
    plt.rc('legend', fontsize=10)
    plt.rc('lines', linewidth=1.5)

def plot_pr(z, var, z_name, var_name, out_path):
    '''
    '''
    plt.figure()
    plt.plot(var, z, '-ko',label=var_name)
    plt.xlabel(var_name, fontsize=14)
    plt.ylabel(z_name, fontsize=14)
    plt.savefig(os.path.join(out_path, 'pr_' + z_name + '_' + var_name + '.png'), bbox_inches='tight')
    plt.close()

def plot_sl(x, y, var, x_name, y_name, var_name, plot_flag, out_path):
    plt.figure()
    if plot_flag == 1:
        plt.pcolormesh(x, y, var.T, cmap='jet')
        # plt.pcolormesh(x, y, var.T, cmap='jet', vmin=-0.001, vmax=0.001)
    elif plot_flag == 2:
        plt.contourf(x, y, var.T, 100, cmap='jet')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.axes().set_aspect('equal')
    plt.colorbar(orientation = 'horizontal', label= var_name, aspect=30)
    plt.savefig(os.path.join(out_path, x_name + y_name + '_'+ var_name + '.png'), bbox_inches='tight')
    plt.close()

def plot_sl_anim(x, y, var, x_name, y_name, var_name, plot_flag, out_path):
    fig = plt.figure()
    ims = []
    for i in range (np.size(var, axis=0)):
        if i == 0:
            if plot_flag == 1:
                # im =plt.pcolormesh(x, y, var[i,:,:].T, cmap='jet')
                im =plt.pcolormesh(x, y, var[i,:,:].T, cmap='jet')
            elif plot_flag == 2:
                im =plt.contourf(x, y, var[i,:,:].T, 100, cmap='jet')
            plt.axes().set_aspect('equal')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.colorbar(orientation = 'horizontal', label= var_name, aspect=30)
        else:
            if plot_flag == 1:
                # im =plt.pcolormesh(x, y, var[i,:,:].T, cmap='jet')
                im =plt.pcolormesh(x, y, var[i,:,:].T, cmap='jet')
            elif plot_flag == 2:
                im =plt.contourf(x, y, var[i,:,:].T, 100, cmap='jet')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=100)
    ani.save(os.path.join(out_path, x_name + y_name + '_'+ var_name + '.gif'), fps=50)
    plt.close()

####################
# PLOT LOG
####################

def plot_log(time, log, config, out_path):
    '''
    '''
    plt.figure()
    plt.plot(time['t'], log['ustar'], '-k')
    plt.plot(time['t'], config['u_fric']*np.ones(time['t'].shape), '--r')
    plt.xlabel('t [s]', fontsize=14)
    plt.ylabel(r'$u_{*} m/s]$', fontsize=14)
    plt.savefig(os.path.join(out_path, 'log_ustar.png'), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(time['t'], log['umax'], '-k')
    plt.xlabel('t [s]', fontsize=14)
    plt.ylabel(r'$u_{max} [m/s]$', fontsize=14)
    plt.savefig(os.path.join(out_path, 'log_umax.png'), bbox_inches='tight')
    plt.close()

####################
# PLOT TURB
####################

def plot_turb(time, turb, config, out_path):
    '''
    '''
    plt.figure()
    plt.plot(time['t'], turb['thrust'], '-k')
    plt.xlabel('t [s]', fontsize=14)
    plt.ylabel(r'$thrust [N]$', fontsize=14)
    plt.savefig(os.path.join(out_path, 'turb_thrust.png'), bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(time['t'], turb['power'], '-k')
    plt.xlabel('t [s]', fontsize=14)
    plt.ylabel(r'$power [W]$', fontsize=14)
    plt.savefig(os.path.join(out_path, 'turb_power.png'), bbox_inches='tight')
    plt.close()


####################
# PLOT ABL
####################

def plot_pr_uvw(space, result_pr, config, out_path):
    '''
    '''
    plt.figure(figsize=(3*3,2*4))
    plt.clf()
    plot_option()
    plt.subplots_adjust(hspace=0.2)

    ax = plt.subplot(231)
    plt.plot(result_pr['u_avg_c']/config['u_fric'], space['z_c']/config['l_z'], 'k')
    plt.xlabel(r'$\bar{u}/u_*$')
    plt.ylabel(r'$z/H$')
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(b=True, which='both')

    ax = plt.subplot(232)
    plt.plot(result_pr['v_avg_c']/config['u_fric'], space['z_c']/config['l_z'], 'g')
    plt.xlabel(r'$\bar{v}/u_*$')
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(b=True, which='both')

    ax = plt.subplot(233)
    plt.plot(result_pr['w_avg_n']/config['u_fric'], space['z_n']/config['l_z'], 'b')
    plt.xlabel(r'$\bar{w}/u_*$')
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(b=True, which='both')

    ax = plt.subplot(234)
    plt.plot(result_pr['u_std_c']/config['u_fric']**2, space['z_c']/config['l_z'], 'k')
    plt.xlabel(r'$\sigma^2_{u}/u^2_*$')
    plt.ylabel(r'$z/H$')
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(b=True, which='both')

    ax = plt.subplot(235)
    plt.plot(result_pr['v_std_c']/config['u_fric']**2, space['z_c']/config['l_z'], 'g')
    plt.xlabel(r'$\sigma^2_{v}/u^2_*$')
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(b=True, which='both')

    ax = plt.subplot(236)
    plt.plot(result_pr['w_std_n']/config['u_fric']**2, space['z_n']/config['l_z'], 'b')
    plt.xlabel(r'$\sigma^2_{w}/u^2_*$')
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(b=True, which='both')

    plt.subplots_adjust(wspace=0.4)

    plt.savefig(os.path.join(out_path, 'pr_uvw.png'), bbox_inches='tight')
    plt.close()

def plot_pr_log(space, result_pr, config, out_path):
    plt.figure()
    plt.semilogy(result_pr['u_avg_c']/config['u_fric'], space['z_c']/config['l_z'], '-ko', fillstyle='none')
    plt.semilogy(1/0.4*np.log(space['z_c']/config['zo']), space['z_c']/config['l_z'], '--r')
    plt.xlabel(r'$\bar{u}/u_*$', fontsize=14)
    plt.ylabel(r'$z/H$', fontsize=14)
    plt.savefig(os.path.join(out_path, 'pr_log.png'), bbox_inches='tight')
    plt.close()

def plot_pr_phi(space, result_pr, config, out_path):
    plt.figure()
    plt.plot(result_pr['dudz_avg_n'][1:]*(space['z_n'][1:]/config['z_i'])*0.4/config['u_fric'], space['z_n'][1:]/config['l_z'], '-ko', fillstyle='none')
    plt.xlim([0.0, 2.0])
    plt.ylim([0.0, 0.6])
    plt.xlabel(r'$\phi$', fontsize=14)
    plt.ylabel(r'$z/H$', fontsize=14)
    plt.savefig(os.path.join(out_path, 'pr_phi.png'), bbox_inches='tight')
    plt.close()

def plot_pr_st(space, result_pr, config, out_path):
    plt.figure()
    plt.plot(result_pr['uw_cov_n']/config['u_fric']**2, space['z_n']/config['l_z'], '-go',label='res', fillstyle='none')
    plt.plot(result_pr['txz_avg_n']/config['u_fric']**2, space['z_n']/config['l_z'], '-bo',label='sgs', fillstyle='none')
    plt.plot((result_pr['uw_cov_n'] + result_pr['txz_avg_n'])/config['u_fric']**2, space['z_n']/config['l_z'], '-ko',label='tot', fillstyle='none')
    plt.xlabel(r'$Norm. stress$', fontsize=14)
    plt.ylabel(r'$z/H$', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(out_path, 'pr_st.png'), bbox_inches='tight')
    plt.close()


####################
# SAVE
####################

def save_pr(z, var, pr_name, out_path):
    R_header = '# ' + pr_name
    R_arrray = np.vstack((z, var)).T
    np.savetxt(os.path.join(out_path, 'pr_' + pr_name + '.txt'), R_arrray, delimiter=' ', fmt='%.4f', header=R_header)
