import numpy as np

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