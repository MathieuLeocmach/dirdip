import numpy as np

def bin_texture(pos, pairs, grid):
    """bin texture tensor on a grid
    
    Parameters
    ----------
    pos: (P,D) array of coordinates
    pairs: (B,2) array of indices defining bounded pairs of particles
    grid: a D-dimentional Grid instance that performs the binning
    
    Returns
    ----------
    sumw: the sum of the texture matrices on each grid element.
    count: the number of matrices binned in each grid element. Each end of a bond counts for 1. The middle of a bond also counts for 1.
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

def bin_geometrical_changes(pos0, pos1, pairs, grid):
    """bin on a grid geometrical changes of the texture tensor between two times. It is based on links which exist at both times.
    
    Parameters
    ----------
    pos0, pos1: (P,D) arrays of coordinates
    pairs: (B,2) array of indices defining bounded pairs of particles at both times
    grid: a D-dimentional Grid instance that performs the binning
    
    Returns
    ----------
    sumw: the sum of the B matrices on each grid element.
    count: the number of matrices binned in each grid element. Each end of a bond that remains on the same grid element between t0 and t1 counts for 1. The middle of a bond also counts for 1 if it emains on the same grid element between t0 and t1. Caution: intensive matrix B is obtained by dividing sumw of the present function by the count of bin_texture (averaged between t0 and t1).
    """
    assert pos0.shape[1] == grid.ndim
    assert pos0.shape[0] == pos1.shape[0]
    assert pos0.shape[1] == pos1.shape[1]
    a = pos0[pairs[:,0]]
    b = pos0[pairs[:,1]]
    bonds0 = b - a
    c = pos1[pairs[:,0]]
    d = pos1[pairs[:,1]]
    bonds1 = d - c
    #average bond and difference
    bonds = 0.5*(bonds0 + bonds1)
    delta_bonds = bonds1 - bonds0
    #extensive version of the asymmetric tensor $C = \ell \otimes \Delta\ell$ see equation C.7
    C = bonds[:,None,:] * delta_bonds[:,:,None]
    #symmetric tensor B is twice the symmetric part
    B = C + C.T
    #since B is symmetric keep only the upper triangle
    i,j = np.triu_indices(pos.shape[1])
    B = B[:,i,j]
    #bin on each end of each bond and on the middle point
    #only points that stay in the same bin will be counted
    sumw = np.zeros(grid.shape+(m.shape[1],))
    count = np.zeros(grid.shape, np.int64)
    for p,q in [(a,c), (b,d), (0.5*(a + b), 0.5*(c + d))]:
        su, co = grid.count_sum_discreet(p, B, q)
        sumw += su
        count += co
    return sumw, count
    