import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection, LineCollection
from .texture import tri2square

def display_pos_edges(ax, pos, pairs, kw_pos = {'marker': 'o', 'c': 'k'}, kw_edges = {'color': 'r'}):
    """Display on a matplotlib axis positions and bonds between them"""
    lc = LineCollection(pos[pairs], **kw_edges)
    ax.add_collection(lc)
    ax.scatter(*pos.T, **kw_pos)
    
def display2Dcount(ax, grid, count, **kw_imshow):
    """Display in a matplotlib axis the result of grid.count for a 2D grid"""
    return ax.imshow(
        np.rot90(count), 
        extent=(
            grid.offsets[0], grid.offsets[0]+grid.steps[0]*(grid.nsteps[0]-1), 
            grid.offsets[1], grid.offsets[1]+grid.steps[1]*(grid.nsteps[1]-1)
        ),
        **kw_imshow,
    )
    
def display_matrices(ax, grid, texture, scale = None):
    """Display on a matplotlib axis an ellipse representing a symmetric matrix at each grid element. Each axis of the ellipse corresponds to an eigenvalue and is oriented along its eigenvector. An axis corresponding to a positive eigenvalue is drawn. A 'coffee bean' has a negative eigenvalue smaller in absolute value than its positive eigenvalue. A 'capsule' has a negative eigenvalue larger in absolute value than its positive eigenvalue. A circle is when the two eigenvalues are equal in absolute value."""
    XY = grid.mesh()
    mask = grid.mask()
    #convert the texture back to an array of matrices
    mat = tri2square(texture)
    #compute egenvalues and eigenvectors of texture for each cell of the grid
    evalues, evectors = np.linalg.eigh(mat)
    #width and height are the larger and smaller eigenvalues respectively
    ww = evalues[...,1][mask]
    hh = evalues[...,0][mask]
    #angle is given by the angle of the larger eigenvector
    aa = np.rad2deg(np.arctan2(evectors[...,1,1], evectors[...,0,1]))[mask]
    #sum of the eigenvalues (trace of the matrix)
    trace = ww+hh#np.where(np.abs(ww)>np.abs(hh), ww, hh)#ww*hh
    #color
    col = plt.cm.viridis((trace - trace.min())/trace.ptp())
    
    if scale is None: 
        #scale = 1
        ellipse_areas = np.pi * np.abs(np.prod(evalues, axis=-1))
        rarea = ellipse_areas / grid.areas()
        scale = np.nanpercentile(np.sqrt(rarea[mask]), 90)
        if scale == 0:
            scale = np.nanmax(np.sqrt(rarea[mask]))
            if scale == 0:
                raise ValueError("All matrices are null")
        scale = 1/scale
        
    
    #show ellipses
    ec = EllipseCollection(
        ww*scale, hh*scale, aa, units='x', offsets=XY,
        transOffset=ax.transData, 
        edgecolors=col, facecolors='none',
    )
    ec.set_array(trace)
    ax.add_collection(ec)
    #major and minor axes (only for positive eigenvalues)
    xyps = scale * np.transpose(evectors*np.maximum(0, evalues)[...,None,:], (0,1,3,2))[mask].reshape(2*len(ww),2)*0.5
    ma = LineCollection(
        [[-xyp, xyp] for xyp in xyps],
        offsets=np.repeat(XY, 2, axis=0),
        color=(0.5,0.5,0.5)
    )
    ax.add_collection(ma)
    return ec

def set_ax_lims(ax, grid):
    """Set x and y limits according to the grid"""
    ax.set_xlim(*grid.low_high_edges(0))
    ax.set_ylim(*grid.low_high_edges(1))
    
def display_polar_grid(ax,grid, color="b"):
    """Show the limits for the polar grid cells"""
    for r in grid.radii:
        ax.add_artist(plt.Circle((0,0), r, fc='none', ec=color))
    for rs, nc, ot in zip(np.column_stack((grid.radii[:-1], grid.radii[1:])), grid.ncells, grid.theta_offset):
        if nc==1:continue
        for theta in 2*np.pi*np.arange(nc)/nc - ot:
            ax.add_artist(plt.Line2D(np.cos(theta)*rs, np.sin(theta)*rs, c=color))