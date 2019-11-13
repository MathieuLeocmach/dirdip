#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:48:39 2019

@author: nklongvessa
"""

from texture.grid import RegularGrid
from texture.texture import bin_texture, bin_changes, bonds_appeared_disapeared
from texture.display import display2Dcount, display_matrices, display_pos_edges
import os
import trackpy as tp
import  numpy as np
from scipy.spatial import Voronoi
import matplotlib.pylab as plt

def voro_edges(pos):
    vor = Voronoi(pos) # create Voronoi diagram
    points_adj = vor.ridge_points
    edges = np.sort(points_adj, axis=-1)
    return np.array(sorted((a,b) for a,b in edges.astype(int)))


path = "/data4To/Ong/Tracking result/19_04_17 Dropping Glass Beads"
path_traj = os.path.join(path,'tracking/around GB_refined/Traj_f_Act{}.h5')

pos_columns = ['x','y']

act = 1
f0 = np.arange(177,187)
lg = 1

grid = RegularGrid([110,200], [30,30], [10,10])
Mtot = np.zeros(grid.shape+(3,))
Ntot = np.zeros(grid.shape, np.int64)
Btot = np.zeros_like(Mtot)
Ttot = np.zeros_like(Mtot)
Na_tot = np.zeros_like(Ntot)
Nd_tot = np.zeros_like(Ntot)
Nc_tot = np.zeros_like(Ntot)


for f in f0:
    
     # get two frames
    with tp.PandasHDFStore(path_traj.format(act)) as s:
        frame0 = s.get(f)
        frame1 = s.get(f + lg)
    
    joined = frame0.join(frame1.set_index('particle'), on ="particle", how='inner', lsuffix='0', rsuffix='1')
    
    pos0 = joined[['x0', 'y0']].to_numpy()
    pos1 = joined[['x1', 'y1']].to_numpy()
        
    #par_intersect = np.intersect1d(frameA.particle,frameB.particle) # particle that exist in both frames
    
     # Voronoi neighbors
    edges0 = voro_edges(pos0)
    edges1 = voro_edges(pos1)
    
    #bin static texture
    for pos, edges in [(pos0, edges0), (pos1, edges1)]:
        sumw, count = bin_texture(pos, edges, grid)
        Mtot += sumw
        Ntot += count
    
    #bin changes between the two frames
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges0, edges1, grid)
    Btot += sumB
    Ttot += sumT
    Na_tot += counta
    Nd_tot += countd
    Nc_tot += countc
    
#from sums to averages
Ntot_nz = 1/np.minimum(1, Ntot)
    
M = Mtot * Ntot_nz[...,None]
B = 2 * Btot * Ntot_nz[...,None]
T = 2 * Ttot * Ntot_nz[...,None]

na = 2 * Na_tot * Ntot_nz
nd = 2 * Nd_tot * Ntot_nz
nc = 2 * Nc_tot * Ntot_nz

#display things!
fig, axs = plt.subplots(2,3, sharex=True, sharey=True, subplot_kw={'aspect':'equal'})
display_matrices(axs[0,0], grid, M, 1e-2)
display_matrices(axs[0,1], grid, B, 1e-2)
display_matrices(axs[0,2], grid, T, 1e-2)
display2Dcount(axs[1,0], grid, nd)
display2Dcount(axs[1,1], grid, na)
display2Dcount(axs[1,2], grid, nc)

plt.show()