'''
Created on 02.07.2019

@author: Mou Lin (moulin1024@gmail.com)

--------------------------------------------------------------------------------
app: create self-explained hdf5 output file
--------------------------------------------------------------------------------
'''

################################################################################
# IMPORT
################################################################################
import os, sys
import fctlib
import app_post as post
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import os.path
import math
import pandas as pd
from pathlib import Path
from matplotlib.pyplot import figure
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
# import fatigue
# from pyevtk.hl import gridToVTK


################################################################################
# MAIN FONCTION
################################################################################
def anime(PATH, case_name):
    '''
    DEF:    post-processing for wireles.
    INPUT:  - case_name
    OUTPUT: - Statistics field: stat.h5
            - Instantanous field: animation.h5 
    '''
    case_path = fctlib.get_case_path(PATH, case_name)

    ############################################################################
    # INIT
    out_path = os.path.join(case_path, 'output')
    in_path = os.path.join(case_path, 'input')
    fctlib.test_and_mkdir(out_path)
    src_out_path = os.path.join(PATH['job'], case_name, 'src', 'output')
    src_inp_path = os.path.join(PATH['job'], case_name, 'src', 'input')

    ############################################################################
    # CONFIG
    print('extract config...')
    config = fctlib.get_config(case_path)

    ############################################################################
    # COMPUTE
    print('compute results...')
    print('Flow Fields:')
    print('Flow Fields:')

    space = post.get_space(config)
    time = post.get_time(config)

    
    if config['ta_flag'] > 0:
        result_3d = post.get_result_3d(src_inp_path,src_out_path, config)
        u_avg = result_3d['u_avg_c']
        v_avg = result_3d['v_avg_c']
        w_avg = result_3d['w_avg_c']

        u_std = result_3d['u_std_c']
        v_std = result_3d['v_std_c']
        w_std = result_3d['w_std_c']

        f = h5py.File(out_path+'/'+case_name+'_stat.h5','w')
        for key, value in config.items():
            f.attrs[key] = value

        f.create_dataset('x',data=space['x'])
        f.create_dataset('y',data=space['y'])
        f.create_dataset('z',data=space['z_c'])

        f.create_dataset('u_avg',data=u_avg )
        f.create_dataset('v_avg',data=v_avg)
        f.create_dataset('w_avg',data=w_avg)

        f.create_dataset('u_std',data=u_std)
        f.create_dataset('v_std',data=v_std)
        f.create_dataset('w_std',data=w_std)

        f.close


    if config['ts_flag'] > 0:

        # f = h5py.File(out_path+'/'+case_name+'_flowfield.h5','w')
        # for key, value in config.items():
        #     f.attrs[key] = value
        result_4d = post.get_result_4d(src_out_path, config)
        # f.create_dataset('x',data=space['x'])
        # f.create_dataset('y',data=space['y'])
        # f.create_dataset('z',data=space['z_c'])

        # f.create_dataset('t_sample',data=time['t_ts'])
        # f.create_dataset('u',data=result_4d['u_inst_c'])
        # f.create_dataset('v',data=result_4d['v_inst_c'])
        # f.create_dataset('w',data=result_4d['w_inst_c'])
        # f.create_dataset('q',data=result_4d['q_inst_c'])

        # f.close

        u = result_4d['u_inst_c']
        # v = result_4d['v_inst_c']
        # w = result_4d['w_inst_c']
        # q = result_4d['q_inst_c']

        # fig,ax = plt.subplots(1,1)
        fig = figure(figsize=(8,8),dpi=300)
        ax1 = fig.add_subplot(111)
        hub = [256/8,896/8]

        # dx = 8
        # dy = 8
        # # ax2 = fig.add_subplot(212)
        def animate(i):  
            print(i)
            # values = u[i,:,:,45]
            ax1.cla()
            # ax2.cla()
            # ax3.cla()

            # ax1.imshow(u[i,:,:,45].T,origin='lower',aspect=config['dx']/config['dy'])
            # ax1.set_xlabel('x')
            # ax1.set_ylabel('y')

            # ax2.imshow(u[i,32,:,:].T,origin='lower',aspect=config['dz']/config['dy'])
            # ax2.set_xlabel('x')
            # ax2.set_ylabel('y')
            # ax2.set_ylim([0,128])
            # print(u[i,128,128,7])
            im = ax1.imshow(u[i,:,:,6].T,origin='lower',extent=[0,10.24,0,5.12])
            # if (i==9):
                # fig.colorbar(im)
            # ax1.quiver(u[i,:,:,45].T,v[i,:,:,45].T)
            # ax1.quiver(space['x'],space['y'],u[i,:,:,90].T,v[i,:,:,90].T,scale=10000)
            ax1.set_xlabel('x (km)')
            ax1.set_ylabel('y (km)')
            # ax1.axis('scaled') 
            ax1.set_ylim([0,5.12])
            ax1.set_xlim([2,8])

            # ax2.plot(m_flap[:i*100,0,0,0])
            # print(i,np.mean(u[i,(224-32):(224-16),(128-16):(128+16),89].flatten()))
            return
        anim = animation.FuncAnimation(fig, animate, frames=10)
        anim.save(out_path+'/animation_xz.gif',writer='pillow', fps=10)

    if config['turb_flag'] > 10:
        turb_loc = pd.read_csv(case_path+"/input/turb_loc.dat")
        f = h5py.File(out_path+'/'+case_name+'_force.h5','w')
        for key, value in config.items():
            f.attrs[key] = value
        f.create_dataset('turb_x',data=turb_loc['x'].to_numpy())
        f.create_dataset('turb_y',data=turb_loc['y'].to_numpy())
        f.create_dataset('turb_z',data=turb_loc['z'].to_numpy())
        f.create_dataset('yaw',data=turb_loc['yaw'].to_numpy())
        f.create_dataset('tilt',data=turb_loc['yaw'].to_numpy())
        
        turb_force = post.get_turb(src_out_path, config)
        f.create_dataset('time',data=time['t'][::2])
        # f.create_dataset('fx',data=turb_force['fx'][::,:,:,:])
        # f.create_dataset('ft',data=turb_force['ft'][::,:,:,:])
        f.create_dataset('displacement_flap',data=turb_force['displacement_flap'][::2,:,:,:])
        f.create_dataset('displacement_edge',data=turb_force['displacement_edge'][::2,:,:,:])
        f.create_dataset('moment_flap',data=turb_force['moment_flap'][::2,:,:,:])
        f.create_dataset('moment_edge',data=turb_force['moment_edge'][::2,:,:,:])
        # f.create_dataset('velocity_flap',data=turb_force['velocity_flap'])
        # f.create_dataset('velocity_edge',data=turb_force['velocity_edge'])
        f.create_dataset('phase',data=turb_force['phase'][::2,:])

        f.close

        m_flap = turb_force['moment_flap']
        plt.figure()
        print(m_flap[:,0,0,0])
        plt.plot(m_flap[:,0,0,0],lw=1)
        # plt.xlim([10000,190000])
        # plt.plot(m_flap[:,0,0,1],lw=0.1,alpha=0.5)
        # plt.plot(m_flap[:,0,0,2],lw=0.1,alpha=0.5)
        print(out_path)
        plt.savefig(out_path+'/flapwise_moment.png')


        # fig = figure(figsize=(14,8))
        # gs = fig.add_gridspec(2, 3)
        # ax1 = fig.add_subplot(gs[0, 0])
        # ax2 = fig.add_subplot(gs[0, 1])
        # ax3 = fig.add_subplot(gs[0, 2])
        # ax4 = fig.add_subplot(gs[1, :])
        # # ax2 = fig.add_subplot(212)
        # def animate(i):  
              
        #     # values = u[i,:,:,45].T
        #     ax1.cla()
        #     ax2.cla()
        #     ax3.cla()
        #     ax4.cla()
        #     # ax3.cla()

        #     ax1.imshow(u[i,:,:,90].T,origin='lower',aspect=config['dx']/config['dy'])
        #     ax1.set_xlabel('x')
        #     ax1.set_ylabel('y')

        #     ax2.imshow(u[i,:,64,:].T,origin='lower',aspect=config['dz']/config['dx'])
        #     ax2.set_xlabel('x')
        #     ax2.set_ylabel('z')

        #     ax3.imshow(u[i,256,:,:].T,origin='lower',aspect=config['dz']/config['dy'])
        #     ax3.set_xlabel('y')
        #     ax3.set_ylabel('z')

        #     ax4.plot(m_flap[:i*100,0,0,0])
        #     print(i)
        #     return
        # anim = animation.FuncAnimation(fig, animate, frames=10)
        # anim.save(out_path+'/animation_xz.gif',writer='pillow', fps=10)


        # downsample = 10
        # flap_scale = 1
        # edge_scale = 40
        
        # initial_phase = [0,-2*np.pi/3,-4*np.pi/3]
        # blade_coord = np.zeros([32,3,3])
        # blade_coord_tilted = np.zeros([32,3,3])
        # blade_coord_yawed = np.zeros([32,3,3])
        # blade = np.linspace(1.5,63,32)
        # omega = 12.1/69*2*np.pi
        # phase_angle = 0
        # tilt_angle = np.deg2rad(df['tilt'][0])
        # yaw_angle = np.deg2rad(df['yaw'][0])

        # print(u.shape)

    # for key, value in config.items():
    #     f.attrs[key] = value
    # f.close

    #     f = h5py.File(out_path+'/'+case_name+'.h5', 'r')
        # print(f.attrs['dx'])

        # for key, value in config.items():
        #     dset.attrs[]
        # dset = f.create_dataset('config',data=config)
       
        # dset = f.create_dataset('v',data=v)
        # dset = f.create_dataset('w',data=w)
        # dset.attrs['property'] = config
        # f.close()
        # print(dset.name)

        # fig = figure(figsize=(8,8),dpi=200)
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224,projection='3d')

        # def animate(i):

        #     ax1.cla() # clear the previous image
        #     ax2.cla() # clear the previous image
        #     ax3.cla() # clear the previous image
        #     ax4.cla() # clear the previous image
        #     phase_angle = (i+1)*downsample*omega*config['dtr']
        #     for j in range(3):
        #         blade_coord[:,j,0] = displacement_flap[(i+1)*downsample,j,:,0]*flap_scale
        #         blade_coord[:,j,1] = blade*np.cos(phase_angle+initial_phase[j]) - displacement_edge[(i+1)*downsample,j,:,0]*edge_scale*np.sin(phase_angle+initial_phase[j])
        #         blade_coord[:,j,2] = blade*np.sin(phase_angle+initial_phase[j]) + displacement_edge[(i+1)*downsample,j,:,0]*edge_scale*np.cos(phase_angle+initial_phase[j])
                
        #         blade_coord_tilted[:,j,0] = blade_coord[:,j,0]*np.cos(tilt_angle) + blade_coord[:,j,2]*np.sin(tilt_angle)
        #         blade_coord_tilted[:,j,1] = blade_coord[:,j,1]
        #         blade_coord_tilted[:,j,2] = blade_coord[:,j,2]*np.cos(tilt_angle) - blade_coord[:,j,0]*np.sin(tilt_angle)
                
        #         blade_coord_yawed[:,j,0] = blade_coord_tilted[:,j,0]*np.cos(yaw_angle) - blade_coord_tilted[:,j,1]*np.sin(yaw_angle)  
        #         blade_coord_yawed[:,j,1] = blade_coord_tilted[:,j,1]*np.cos(yaw_angle) + blade_coord_tilted[:,j,0]*np.sin(yaw_angle) 
        #         blade_coord_yawed[:,j,2] = blade_coord_tilted[:,j,2] 


        #         blade_coord_yawed[:,j,0] = turb_x + blade_coord_yawed[:,j,0]
        #         blade_coord_yawed[:,j,1] = turb_y + blade_coord_yawed[:,j,1]
        #         blade_coord_yawed[:,j,2] = turb_z + blade_coord_yawed[:,j,2]

        #         ax1.plot(blade_coord_yawed[:,j,1], blade_coord_yawed[:,j,2])
        #         ax2.plot(blade_coord_yawed[:,j,0], blade_coord_yawed[:,j,2])
        #         ax3.plot(blade_coord_yawed[:,j,0], blade_coord_yawed[:,j,1])
        #         ax4.plot3D(blade_coord_yawed[:,j,0],blade_coord_yawed[:,j,1], blade_coord_yawed[:,j,2])

        #     ax1.axis('scaled')
        #     ax1.set_xlabel('y (m)')
        #     ax1.set_ylabel('z (m)')
        #     ax1.set_ylim([0,200]) # fix the x axis
        #     ax1.set_xlim([turb_y-100,turb_y+100]) # fix the y axis
            
        #     ax2.axis('scaled')
        #     ax2.set_xlabel('x (m)')
        #     ax2.set_ylabel('z (m)')
        #     ax2.set_ylim([0,200]) # fix the x axis
        #     ax2.set_xlim([turb_x-100,turb_x+100]) # fix the y axis

        #     ax3.axis('scaled')
        #     ax3.set_xlabel('x (m)')
        #     ax3.set_ylabel('y (m)')
        #     ax3.set_ylim([turb_y-100,turb_y+100]) # fix the x axis
        #     ax3.set_xlim([turb_x-100,turb_x+100]) # fix the y axis

        #     ax4.set_xlabel('x (m)')
        #     ax4.set_ylabel('y (m)')
        #     ax4.set_zlabel('z (m)')
        #     ax4.set_zlim([0,200]) # fix the x axis
        #     ax4.set_xlim([turb_x-100,turb_x+100]) # fix the y axis
        #     ax4.set_ylim([turb_y-100,turb_y+100]) # fix the y axis

        #     print(i)

        # anim = animation.FuncAnimation(fig, animate, frames=10)
        # anim.save(out_path+'/blade_movement_3d.gif',writer='pillow', fps=10)