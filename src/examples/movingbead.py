#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:48:39 2019

@author: nklongvessa
"""

from texture.grid import RegularGrid
from texture.texture import bin_texture, bin_changes, bonds_appeared_disapeared, C2B
from texture.display import display2Dcount, display_matrices, display_pos_edges
import os
import trackpy as tp
import  numpy as np
from scipy.spatial import Voronoi
import matplotlib.pylab as plt
import pandas as pd

plt.rcParams['image.cmap'] = 'viridis'

def voro_edges(pos, l):
    vor = Voronoi(pos) # create Voronoi diagram
    points_adj = vor.ridge_points
    edges = np.sort(points_adj, axis=-1)
    
    #remove too long edges
    r1 = pos[edges[:,0]]
    r2 = pos[edges[:,1]]
    edges_length = np.linalg.norm(r1-r2,axis = 1)
    keep = edges_length < l    
    edges_keep = edges[keep]
   
    return np.array(sorted((a,b) for a,b in edges_keep.astype(int)))


path = "/data4To/Ong/Tracking result/19_04_17 Dropping Glass Beads"
path_traj = os.path.join(path,'tracking/around GB_refined/Traj_f_Act{}.h5')
path_trajGB = os.path.join(path,'tracking/GB motion/Traj_f_Act{}.csv')

pos_columns = ['x','y']

select_act = [1]#,4]#4#1
lg = 1
bonds_max_len = 20 # threshold of edge length (recommened 1.5 diameter)
f0_min = [80+27,0+33]
f0_len = [20,76]#[40,76]

for ind, act in enumerate(select_act):
    traj_GB = pd.read_csv(path_trajGB.format(act))
    f0 = traj_GB.frame[f0_min[ind]:f0_min[ind] + f0_len[ind] +lg]#traj_GB.frame[80:150]#traj_GB.frame[42:142]#np.arange(177,177+50)
    
    # glass bead's traj
    
    
    # initialize metrices
    grid = RegularGrid([-230,-230], [20,20], [23,23])
    Mtot = np.zeros(grid.shape+(3,))
    Ntot = np.zeros(grid.shape, np.int64)
    Ctot = np.zeros(grid.shape+(2,2))
    Ttot = np.zeros_like(Mtot)
    Na_tot = np.zeros_like(Ntot)
    Nd_tot = np.zeros_like(Ntot)
    Nc_tot = np.zeros_like(Ntot)
    
    
    for indf, f in enumerate(f0[:-lg]):
        
        print('Progress: {}%'.format(indf*100/len(f0)))
        
        # glass beads' position
        posGB0 = traj_GB[traj_GB.frame == f0.iloc[indf]][['x','y']].to_numpy()
        posGB1 = traj_GB[traj_GB.frame == f0.iloc[indf+1]][['x','y']].to_numpy()
        
       
        
        # get two frames
        with tp.PandasHDFStore(path_traj.format(act)) as s:
            frame0 = s.get(f)
            frame1 = s.get(f + lg)
        
        joined = frame0.join(frame1.set_index('particle'), on ="particle", how='inner', lsuffix='0', rsuffix='1')
        
        pos0 = joined[['x0', 'y0']].to_numpy() - posGB0
        pos1 = joined[['x1', 'y1']].to_numpy() - posGB1
        
        pos0 = pos0[:,::-1] * [1,-1]
        pos1 = pos1[:,::-1] * [1,-1]
            
        #par_intersect = np.intersect1d(frameA.particle,frameB.particle) # particle that exist in both frames
        
         # Voronoi neighbors
        edges0 = voro_edges(pos0, bonds_max_len)
        edges1 = voro_edges(pos1, bonds_max_len)
        
    
        
        
        #bin static texture
        for pos, edges in [(pos0, edges0), (pos1, edges1)]:
            sumw, count = bin_texture(pos, edges, grid)
            Mtot += sumw
            Ntot += count
        
        #bin changes between the two frames
        sumC, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges0, edges1, grid)
        Ctot += sumC
        Ttot += sumT
        Na_tot += counta
        Nd_tot += countd
        Nc_tot += countc
        
    #from sums to averages
    Ntot_nz = 1/np.maximum(1, Ntot)
    mask = Ntot > 500
        
    M = Mtot * Ntot_nz[...,None]
    C = 2 * Ctot * Ntot_nz[...,None, None]
    T = 2 * Ttot * Ntot_nz[...,None]
    B = C2B(C)
    
    na = 2 * Na_tot * Ntot_nz
    nd = 2 * Nd_tot * Ntot_nz
    nc = 2 * Nc_tot * Ntot_nz
    
    #display things!
    fig, axs = plt.subplots(3,3, figsize=(10,10),sharex=True, sharey=True, subplot_kw={'aspect':'equal'})
    fig.canvas.set_window_title('Act {}'.format(act))
    
    display_matrices(axs[0,0], grid, M * mask[...,None])
    axs[0,0].title.set_text('Texture M')
    
    display_matrices(axs[0,1], grid, B * mask[...,None])
    axs[0,1].title.set_text('Geometrical change B')
    
    display_matrices(axs[0,2], grid, T * mask[...,None])
    axs[0,2].title.set_text('Topological change T')
    
    ima = display2Dcount(axs[1,0], grid, nd)
    axs[1,0].title.set_text('Disappeared bonds')
    
    display2Dcount(axs[1,1], grid, na)
    axs[1,1].title.set_text('Appeared bonds')   
    
    ima2 = display2Dcount(axs[1,2], grid, nc)
    fig.colorbar(ima2, ax=axs[1,2:3], location='bottom')
    fig.colorbar(ima, ax=axs[1, :2], shrink=0.6, location='bottom')
    
    #
    axs[1,2].title.set_text('Conserved bonds')
    
    V, Omega, P = statistical_relative_deformations(M, C, T)
    display.display_matrices(axs[2,0], grid, V)
    axs[2,0].title.set_text('Statistical velocity gradient V')
    display.display2Dcount(axs[2,1], grid, Omega, cmap=plt.cm.seismic, vmin=-np.abs(Omega).max(), vmax=np.abs(Omega).max())
    axs[2,1].title.set_text(r'Statistical rotation rate $\Omega$')
    display.display_matrices(axs[2,2], grid, P)
    axs[2,2].title.set_text('Statistical topological rearrangement rate P')


plt.show()
