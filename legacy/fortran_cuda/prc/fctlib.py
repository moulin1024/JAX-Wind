'''
mod on Jan 17, 2019

@author: trevaz (tristan.revaz@epfl.ch)

---------------------------------------------------------------------
fctlib
---------------------------------------------------------------------
'''

#################################################################################
# IMPORTS
#################################################################################
import os
import sys
import numpy as np

#################################################################################
# CONSTANTS
#################################################################################


#################################################################################
# FUNCTIONS
#################################################################################

def zs_fct0(x,y):
    X, Y = np.meshgrid(x,y,indexing='ij')
    zs = np.zeros(X.shape)

    return zs

def zs_fct1(x,y):
    zs_h = 0.285
    zs_l = 0.570
    sigma = zs_l/1.1774
    zs_x = 5.0
    X, Y = np.meshgrid(x,y,indexing='ij')
    zs = zs_h*np.exp(-0.5*((X-zs_x)/sigma)**2)
    # for i in range (x.size):
    #     for j in range (y.size):
    #         if (x[i]-zs_x)**2+(y[j]-zs_y)**2>zs_l**2:
    #             zs[i,j] = 0.

    return zs

def zs_fct2(x,y):
    zs_h = 0.04
    zs_l = 0.1
    zs_x = 0.6
    zs_y = 0.3
    X, Y = np.meshgrid(x,y,indexing='ij')
    zs = zs_h*np.cos( np.pi*( (X-zs_x)**2 + (Y-zs_y)**2 )**0.5 / (2*zs_l) )**2
    for i in range (x.size):
        for j in range (y.size):
            if (x[i]-zs_x)**2+(y[j]-zs_y)**2>zs_l**2:
                zs[i,j] = 0.

    return zs

# def zs_fct2(x,y):
#     zs_h = 0.04
#     zs_l = 0.1
#     zs_x = 0.6
#     X, Y = np.meshgrid(x,y,indexing='ij')
#     zs = zs_h*np.cos( np.pi*( (X-zs_x)**2)**0.5 / (2*zs_l) )**2
#     for i in range (x.size):
#         for j in range (y.size):
#             if (x[i]-zs_x)**2>zs_l**2:
#                 zs[i,j] = 0.
#
#     return zs

# def zs_fct2(x,y):
#     zs_h = 0.04
#     zs_l = 0.1
#     zs_x = 0.6
#     X, Y = np.meshgrid(x,y,indexing='ij')
#     zs = np.zeros(X.shape)
#     for i in range (x.size):
#         for j in range (y.size):
#             if ( ((x[i]-zs_x)<=0) & ((x[i]-zs_x)>=-zs_l) ):
#                 zs[i,j] = zs_h+zs_h/zs_l*(x[i]-zs_x)
#     for i in range (x.size):
#         for j in range (y.size):
#             if ( ((x[i]-zs_x)>=0) & ((x[i]-zs_x)<=zs_l) ):
#                 zs[i,j] = zs_h-zs_h/zs_l*(x[i]-zs_x)
#     return zs
#
# def zs_fct2(x,y):
#     X, Y = np.meshgrid(x,y,indexing='ij')
#     zs = X
#     return zs

def get_case_path(PATH, case_name):
    case_path = os.path.join(PATH['job'], case_name)
    if not os.path.isdir(case_path):
        print('\n --> case does not exist')
        sys.exit()
    return case_path

def parse_file(file_path):
    '''
    DEF:    parse file
    INPUT:  - file_path
    OUTPUT: - ()
    '''
    COMMENT_CHAR = '#'
    OPTION_CHAR =  '='

    file = {}
    f = open(file_path)
    for line in f:
        # First, remove comments:
        if COMMENT_CHAR in line:
            # split on comment char, keep only the part before
            line, comment = line.split(COMMENT_CHAR, 1)
        # Second, find lines with an option=value:
        if OPTION_CHAR in line:
            # split on option char:
            option, value = line.split(OPTION_CHAR, 1)
            # strip spaces:
            option = option.strip()
            value = value.strip()
            # store in dictionary:
            file[option] = value
    f.close()

    return file


def get_config(case_path):
    '''
    DEF:    parse file
    INPUT:  - file_path
    OUTPUT: - ()
    '''

    config = parse_file(os.path.join(case_path, 'input', 'config'))

    # init
    config['sim_flag'] = int(config['sim_flag'])
    if config['sim_flag'] == 1:
        if not os.path.isdir(os.path.join('job', config['interp_case'])):
            print('ERROR: interp case specifed does not exist')
            sys.exit()
    if config['sim_flag'] == 2:
        if not os.path.isdir(os.path.join('job', config['warmup_case'])):
            print('ERROR: warmup case specifed does not exist')
            sys.exit()
    if config['sim_flag'] == 3:
        if not os.path.isdir(os.path.join('job', config['prec_case'])):
            print('ERROR:prec case specifed does not exist')
            sys.exit()

    config['resub_flag'] = int(config['resub_flag'])
    config['double_flag'] = int(config['double_flag'])

    # space
    if int(config['dom_flag']) == 0:
        config['nx'] = int(config['nx'])
        config['ny'] = int(config['ny'])
        config['nz'] = int(config['nz'])
        config['z_i'] = float(config['z_i'])
        config['l_z'] = float(config['l_z'])
        config['l_r'] = int(config['l_r'])

    elif int(config['dom_flag']) == 1:
        config['nx'] = int(config['nx'])
        config['ny'] = int(config['ny'])
        config['nz'] = int(config['nz'])
        config['z_i'] = float(config['lx'])/(2*np.pi)
        config['l_z'] = float(config['lz'])
        config['l_r'] = int(float(config['lx'])/float(config['ly']))

    config = extract_space_param(config)

    # time
    if int(config['time_flag']) == 0:
        config['nsteps'] = int(config['nsteps'])
        config['dt'] = float(config['dt'])

    elif int(config['time_flag']) == 1:
        config['dt'] = float(config['dtr'])/float(config['z_i'])
        config['nsteps'] = int(config['nsteps'])

    config = extract_time_param(config)

    # physic
    config['zo'] = float(config['zo'])
    config['u_fric'] = float(config['u_fric'])
    config['bl_height'] = float(config['bl_height'])

    # sgs
    config['model'] = int(config['model'])
    config['fgr'] = float(config['fgr'])
    config['tfr'] = float(config['tfr'])
    config['cs_count'] = int(config['cs_count'])

    # turbine
    config['turb_flag'] = int(config['turb_flag'])
    config['turb_nb'] = int(config['turb_nb'])
    # config['turb_i'] = config['turb_i']
    # config['turb_j'] = config['turb_j']
    config['turb_r'] = float(config['turb_r'])
    config['turb_z'] = config['turb_z']
    config['turb_w'] = config['turb_w']
    config['tow_r'] = float(config['tow_r'])
    config['tow_c'] = float(config['tow_c'])
    config['nac_r'] = float(config['nac_r'])
    config['nac_c'] = float(config['nac_c'])
    # config['yaw_angle'] = config['yaw_angle']
    config['turb_count'] = int(config['turb_count'])

    # prec
    config['inflow_istart'] = int(config['inflow_istart'])
    config['inflow_iend'] = int(config['inflow_iend'])
    config['inflow_nx'] = config['inflow_iend']-config['inflow_istart']+1
    config['inflow_count'] = int(config['inflow_count'])

    # output
    config['log_flag'] = int(config['log_flag'])
    config['c_count'] = int(config['c_count'])
    config['p_count'] = int(config['p_count'])
    config['ta_flag'] = int(config['ta_flag'])
    config['ta_mask'] = int(config['ta_mask'])
    if config['ta_mask'] == 1:
        config['ta_istart'] = int(config['ta_istart'])
        config['ta_iend'] = int(config['ta_iend'])
        config['ta_jstart'] = int(config['ta_jstart'])
        config['ta_jend'] = int(config['ta_jend'])
        config['ta_kend'] = int(config['ta_kend'])
        config['ta_tstart'] = int(config['ta_tstart'])
    else:
        config['ta_istart'] = 1
        config['ta_iend'] = config['nx']
        config['ta_jstart'] = 1
        config['ta_jend'] = config['ny']
        config['ta_kend'] = config['nz']
    config['ta_tstart'] = int(config['ta_tstart'])
    config['ta_nx'] = config['ta_iend']-config['ta_istart']+1
    config['ta_ny'] = config['ta_jend']-config['ta_jstart']+1
    config['ta_ns'] = int((config['nsteps']-config['ta_tstart']+1)/config['p_count'])
    config['ts_flag'] = int(config['ts_flag'])
    config['ts_mask'] = int(config['ts_mask'])
    if config['ts_mask'] == 1:
        config['ts_istart'] = int(config['ts_istart'])
        config['ts_iend'] = int(config['ts_iend'])
        config['ts_jstart'] = int(config['ts_jstart'])
        config['ts_jend'] = int(config['ts_jend'])
        config['ts_kend'] = int(config['ts_kend'])
        config['ts_tstart'] = int(config['ts_tstart'])
    else:
        config['ts_istart'] = 1
        config['ts_iend'] = config['nx']
        config['ts_jstart'] = 1
        config['ts_jend'] = config['ny']
        config['ts_kend'] = config['nz']
    config['ts_tstart'] = int(config['ts_tstart'])
    config['ts_nx'] = config['ts_iend']-config['ts_istart']+1
    config['ts_ny'] = config['ts_jend']-config['ts_jstart']+1
    config['ts_ns'] = int((config['nsteps']-config['ts_tstart']+1)/(config['c_count']*10))

    # exec
    config['nzb'] = int(config['nz'])//int(config['job_np'])
    config['nz2'] = int(config['nzb'])+2
    config['job_time'] = config['job_time']
    config['job_np'] = int(config['job_np'])

    return config

def extract_space_param(config):
    config['lx'] = config['z_i']*2*np.pi
    config['ly'] = config['z_i']*2*np.pi/config['l_r']
    config['lz'] = config['l_z']
    config['dx'] = config['lx']/config['nx']
    config['dy'] = config['ly']/config['ny']
    config['dz'] = config['lz']/(config['nz']-1)
    return config

def extract_time_param(config):
    config['dtr'] = config['dt']*config['z_i']
    config['t_tot'] = config['nsteps']*config['dtr']
    return config


def get_space(config):
    space = {}
    space['x'] = np.arange(0, config['lx'], config['dx'])
    space['y'] = np.arange(0, config['ly'], config['dy'])
    space['z_n'] = np.arange(0, config['lz']+config['dz'], config['dz'])
    space['z_c'] = node2center_1d(space['z_n'])
    space['x_'] = np.arange(0, config['lx']+config['dx'], config['dx']) - 0.5*config['dx']
    space['y_'] = np.arange(0, config['ly']+config['dy'], config['dy']) - 0.5*config['dy']
    return space


def read_file(file_path):
    '''
    DEF:    read file
    INPUT:  - file_path
    OUTPUT: - file_str
    '''
    file_id = open(file_path, 'r')
    file_str = file_id.read()
    file_id.close()
    return file_str

def write_file(file_str, file_path):
    '''
    DEF:    write file
    INPUT:  - file_str
            - file_path
    OUTPUT: - ()
    '''
    lsf = open(file_path, 'w')
    lsf.write(file_str)
    lsf.close()


def test_and_mkdir(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

def load_1d(var_name, Nf, double_flag, path):
    var = np.fromfile(os.path.join(path, var_name + '.bin'), dtype=np.float64, count=Nf)
    return var

def load_4d(var_name, Nf, Nx, Ny, Nz, double_flag, path):
    if double_flag == 0:
        var = np.fromfile(os.path.join(path, var_name + '.bin'), dtype=np.float32, count=Nf*Nx*Ny*Nz)
    else:
        var = np.fromfile(os.path.join(path, var_name + '.bin'), dtype=np.float64, count=Nf*Nx*Ny*Nz) 
    var = np.transpose(var.reshape((Nx,  Ny,  Nz, Nf), order='F'), (3,0,1,2))
    return var

def load_3d(var_name, Nx, Ny, Nz, double_flag, path):
    if double_flag == 0:
        var = np.fromfile(os.path.join(path, var_name + '.bin'), dtype=np.float32, count=Nx*Ny*Nz)
    else:
        var = np.fromfile(os.path.join(path, var_name + '.bin'), dtype=np.float64, count=Nx*Ny*Nz)        
    var = var.reshape((Nx,  Ny,  Nz), order='F')
    return var

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
    var_n[:,:,:,0] = var_c[:,:,:,0]
    var_n[:,:,:,1:] = 0.5*(var_c[:,:,:,:-1]+var_c[:,:,:,1:])
    return var_n

def center2node_3d(var_c):
    var_n = np.copy(var_c)
    var_n[:,:,0] = var_c[:,:,0]
    var_n[:,:,1:] = 0.5*(var_c[:,:,:-1]+var_c[:,:,1:])
    return var_n
