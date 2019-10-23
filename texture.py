import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection

def bin_texture(pos, pairs, grid):
    """bin texture tensor on a grid
    
    Parameters
    ----------
    pos: (P,D) array of coordinates
    pairs: (B,2) array of indices defining bounded pairs of particles
    grid: a D-dimentional Grid instance that performs the binning
    """
    assert pos.shape[1] == grid.ndim
    a = pos[pairs[:,0]]
    b = pos[pairs[:,1]]
    bonds = b - a
    #link matrix (symmetric, but compute everything, that is a ratio of (D-1)/(2*D) too many coefficients: 1/4, 1/3, etc)
    m = bonds[:,None,:] * bonds[:,:, None]
    #since m is symmetric keep only the upper triangle
    i,j = np.triu_indices(pos.shape[1])
    m = m[:,i,j]
    #bin on each end of each bond and on the middle point
    sumw = np.zeros(grid.shape+(m.shape[1],))
    count = np.zeros(grid.shape, np.int64)
    for p in [a, b, 0.5*(a + b)]:
        s, c = grid.count_sum_discreet(p, m)
        sumw += s
        count += c
    return sumw, count

def display_pos_edges(ax, pos, pairs):
    """Display on a matplotlib axis positions and bonds between them"""
    lc = LineCollection(pos[pairs], color='r')
    ax.add_collection(lc)
    ax.plot(*pos.T,'ok')
    
def display_texture(ax, grid, texture):
    """Display on a matplotlib axis an ellipse representing the texture at each grid element"""
    x,y = np.transpose(grid.offsets + (0.5+np.transpose([np.arange(n) for n in grid.nsteps-1]))*grid.steps)
    #rotate 90° to be consistent with axis orientation
    X, Y = np.meshgrid(x, y[::-1])
    XY = np.column_stack((X.ravel(), Y.ravel()))
    #compute egenvalues and eigenvectors of texture for each cell of the grid
    evalues, evectors = np.linalg.eigh(np.rot90(texture[...,[0,1,1,2]].reshape(texture.shape[:-1]+(2,2))))
    #width and height are the larger and smaller eigenvalues respectively
    ww = evalues[...,1].ravel()
    hh = evalues[...,0].ravel()
    #angle is given by the angle of the larger eigenvector
    aa = np.rad2deg(np.arctan2(evectors[...,1,1], evectors[...,0,1])).ravel()
    
    #show ellipses
    ec = EllipseCollection(
        ww, hh, aa, units='x', offsets=XY,
        transOffset=ax.transData
    )
    ec.set_array(np.rot90(count).ravel())
    ax.add_collection(ec)
    #major and minor axes
    xyps = (evectors*evalues[...,None]).reshape(2*len(ww),2)*0.5
    ma = LineCollection(
        [[-xyp, xyp] for xyp in xyps],
        offsets=np.repeat(XY, 2, axis=0),
        color='y'
    )
    ax.add_collection(ma)