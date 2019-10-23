import numpy as np
from matplotlib.collections import EllipseCollection, LineCollection

def display_pos_edges(ax, pos, pairs):
    """Display on a matplotlib axis positions and bonds between them"""
    lc = LineCollection(pos[pairs], color='r')
    ax.add_collection(lc)
    ax.plot(*pos.T,'ok')
    
def display2Dcount(grid, ax, count):
    """Display in a matplotlib axis the result of grid.count for a 2D grid"""
    ax.imshow(
        np.rot90(count), 
        extent=(
            grid.offsets[0], grid.offsets[0]+grid.steps[0]*(grid.nsteps[0]-1), 
            grid.offsets[1], grid.offsets[1]+grid.steps[1]*(grid.nsteps[1]-1)
        )
    )
    
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