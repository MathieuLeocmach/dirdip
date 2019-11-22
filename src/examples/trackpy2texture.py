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

def voro_edges(pos):
    """Voronoi neighbors."""
    vor = Voronoi(pos) # create Voronoi diagram
    points_adj = vor.ridge_points
    edges = np.sort(points_adj, axis=-1)
    return np.array(sorted((a,b) for a,b in edges.astype(int)))


path = "/data4To/Ong/Tracking result/19_04_17 Dropping Glass Beads"
path_traj = os.path.join(path,'tracking/around GB_refined/Traj_f_Act{}.h5')



act = 1
f0 = np.arange(177,187)
lg = 1

grid = RegularGrid([110,200], [30,30], [10,10])
Mtot = np.zeros(grid.shape+(3,))
Ntot = np.zeros(grid.shape, np.int64)
Ctot = np.zeros(grid.shape+(2,2))
Ttot = np.zeros_like(Mtot)
Na_tot = np.zeros_like(Ntot)
Nd_tot = np.zeros_like(Ntot)
Nc_tot = np.zeros_like(Ntot)


for f in f0:
    
    # get two frames
    with tp.PandasHDFStore(path_traj.format(act)) as s:
        frame0 = s.get(f)
        frame1 = s.get(f + lg)
    
    # work only on particles that exist in both frames
    ## gather in one DataFrame the two times so that in a single row we have the information about the particle at the two time steps
    joined = frame0.join(frame1.set_index('particle'), on ="particle", how='inner', lsuffix='0', rsuffix='1')
    
    # extract only the coordinates at each time step. Thanks to the previous step, we have ensured that the rows are sorted consistently
    pos0 = joined[['x0', 'y0']].to_numpy()
    pos1 = joined[['x1', 'y1']].to_numpy()
    
    # Voronoi neighbors
    ## You may want to refine this with a distance criterion or use another definition of the bonds
    edges0 = voro_edges(pos0)
    edges1 = voro_edges(pos1)
    
    # if the grid is mobile between the two time steps (translation, shear, etc.), that is where you should transform the coordinates
    
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

## matrices
### texture, in length**2
M = Mtot * Ntot_nz[...,None]
### geometrical changes, in length**2/time (probably px**2*fps)
C = 2 * Ctot * Ntot_nz[...,None, None]
B = C2B(C)
### topological changes, in length**2/time (probably px**2*fps)
T = 2 * Ttot * Ntot_nz[...,None]

## densities
### apprearing bonds, in 1/time (probably fps)
na = 2 * Na_tot * Ntot_nz
### disapprearing bonds, in 1/time (probably fps)
nd = 2 * Nd_tot * Ntot_nz
### conserved bonds, in 1/time (probably fps)
nc = 2 * Nc_tot * Ntot_nz

# display things!
fig, axs = plt.subplots(2,3, sharex=True, sharey=True, subplot_kw={'aspect':'equal'})
display_matrices(axs[0,0], grid, M)
display_matrices(axs[0,1], grid, B)
display_matrices(axs[0,2], grid, T)
ima = display2Dcount(axs[1,0], grid, nd)
display2Dcount(axs[1,1], grid, na)
fig.colorbar(display2Dcount(axs[1,2], grid, nc), ax=axs[1, 2], location='bottom')
fig.colorbar(ima, ax=axs[1, :2], shrink=0.6, location='bottom')

plt.show()