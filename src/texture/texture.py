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
    sumC: the sum of the C matrices on each grid element. From C, one obtains `B = C + np.transpose(C, axes=(0,2,1))`. Provided the texture M, we can also obtain V and $\Omega$.
    count: the number of matrices binned in each grid element. Each end of a bond that remains on the same grid element between t0 and t1 counts for 1. The middle of a bond also counts for 1 if it emains on the same grid element between t0 and t1. Caution: intensive matrix C is obtained by dividing sumC of the present function by the count of bin_texture (averaged between t0 and t1).
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
    #since C is not symmetric we have to keep all coefficients
    #bin on each end of each bond and on the middle point
    #only points that stay in the same bin will be counted
    sumw = np.zeros(grid.shape + C.shape[1:])
    count = np.zeros(grid.shape, np.int64)
    for p,q in [(a,c), (b,d), (0.5*(a + b), 0.5*(c + d))]:
        su, co = grid.count_sum_discreet(p, C, q)
        sumw += su
        count += co
    return sumw, count

def C2B(C):
    """convert from a C (or sumC) matrix to a B (resp sumB) matrix (upper triangle)"""
    axes = np.arange(C.ndim)
    axes[-2:] = axes[-2:][::-1]
    B = (C + np.transpose(C, axes=axes))
    i,j = np.triu_indices(C.shape[-1])
    return B[...,i,j]

def bonds_set_to_array(s):
    if len(s)==0:
        return np.zeros((0,2), dtype=np.int64)
    return np.array([[a,b] for a,b in sorted(s)], dtype=np.int64)

def bonds_appeared_disapeared(pairs0, pairs1):
    """Compute the bonds that appeared between two sets of bonds, and the bonds that disappeared, and the bonds that are conserved"""
    s0 = set((a,b) for a,b in np.sort(pairs0, axis=-1))
    s1 = set((a,b) for a,b in np.sort(pairs1, axis=-1))
    pairsa = bonds_set_to_array(s1-s0)
    pairsd = bonds_set_to_array(s0-s1)
    pairsc = bonds_set_to_array(s0&s1)
    return pairsa, pairsd, pairsc
    
def bin_topological_changes(pos0, pos1, pairs0, pairs1, grid):
    """bin on a grid topological changes of the texture tensor between two times. It is based on links which appeared or disappeared between both times.
    
    Parameters
    ----------
    pos0, pos1: (P,D) arrays of coordinates for particles that exist at both times
    pairs0: (B0,2) array of indices defining bounded pairs of particles at t0
    pairs1: (B1,2) array of indices defining bounded pairs of particles at t1
    grid: a D-dimentional Grid instance that performs the binning
    
    Returns
    ----------
    sumw: the sum of the T matrices on each grid element. Caution: intensive matrix T is obtained by dividing sumw of the present function by the count of bin_texture (averaged between t0 and t1).
    counta: the number of appearing matrices binned in each grid element. Each end of an appearing bond at t1 counts for 1. The middle of a bond also counts for 1.
    countd: the number of disappearing matrices binned in each grid element. Each end of a disappearing bond at t0 counts for 1. The middle of a bond also counts for 1.
    """
    assert pos0.shape[1] == grid.ndim
    assert pos0.shape[0] == pos1.shape[0]
    assert pos0.shape[1] == pos1.shape[1]
    #bonds that appeared and disappeared between t0 and t1
    pairsa, pairsd, pairsc = bonds_appeared_disapeared(pairs0, pairs1)
    #bin the texture of bonds that appeared
    sumwa, counta = bin_texture(pos1, pairsa, grid)
    #bin the texture of bonds that disappeared
    sumwd, countd = bin_texture(pos0, pairsd, grid)
    return sumwa-sumwd, counta, countd
    
def bin_changes(pos0, pos1, pairs0, pairs1, grid):
    """bin on a grid geometrical and topological changes of the texture tensor between two times.
    Caution: intensive matrices C and T are obtained by dividing sumC and sumT of the present function by the count of bin_texture (averaged between t0 and t1).
    
    Parameters
    ----------
    pos0, pos1: (P,D) arrays of coordinates for particles that exist at both times
    pairs0: (B0,2) array of indices defining bounded pairs of particles at t0
    pairs1: (B1,2) array of indices defining bounded pairs of particles at t1
    grid: a D-dimentional Grid instance that performs the binning
    
    Returns
    ----------
    sumC: the sum of the C matrices on each grid element.
    countc: the number of matrices binned in each grid element. Each end of a bond that remains on the same grid element between t0 and t1 counts for 1. The middle of a bond also counts for 1 if it emains on the same grid element between t0 and t1.
    sumT: the sum of the T matrices on each grid element.
    counta: the number of appearing matrices binned in each grid element. Each end of an appearing bond at t1 counts for 1. The middle of a bond also counts for 1.
    countd: the number of disappearing matrices binned in each grid element. Each end of a disappearing bond at t0 counts for 1. The middle of a bond also counts for 1.
    """
    assert pos0.shape[1] == grid.ndim
    assert pos0.shape[0] == pos1.shape[0]
    assert pos0.shape[1] == pos1.shape[1]
    #bonds that appeared, disappeared, or were conserved between t0 and t1
    pairsa, pairsd, pairsc = bonds_appeared_disapeared(pairs0, pairs1)
    #bin the texture of bonds that appeared
    sumwa, counta = bin_texture(pos1, pairsa, grid)
    #bin the texture of bonds that disappeared
    sumwd, countd = bin_texture(pos0, pairsd, grid)
    #bin the geometrical changes of the conserved bonds
    sumC, countc = bin_geometrical_changes(pos0, pos1, pairsc, grid)
    return sumC, countc, sumwa-sumwd, counta, countd