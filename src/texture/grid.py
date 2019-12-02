import numpy as np
from numba import jit

@jit(nopython=True)
def digitize_regular(xs, offset, step, nstep):
    """snap coordinates to a regular grid, assigning to it the index of the edge immediately larger to it.
            Coordinates that are below the offset will have index 0.
            Coordinates that are above offset + step * nstep will have index nstep."""
    ipos = np.ceil((xs-offset)/step).astype(np.int64)
    ipos[ipos>nstep] = nstep
    ipos[ipos<0] = 0
    return ipos

@jit(nopython=True)
def bin_count(x, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints.

The number of bins (of size 1) is one larger than the largest value in x. If minlength is specified, there will be at least this number of bins in the output array (though it will be longer if necessary, depending on the contents of x). Each bin gives the number of occurrences of its index value in x.

Parameters
----------
x : array_like, 1 dimension, nonnegative ints
    Input array.
    
minlength : int
    A minimum number of bins for the output array.

Returns
----------
count : 1D ndarray of ints
    The number of elements of x in each bin. The length of count is `max(minlength, np.amax(x)+1)`.

"""
    mx = x.max()
    mn = x.min()
    assert mn >= 0
    ans_size = max(mx+1, minlength)
    ans = np.zeros((ans_size,), dtype=np.int64)
    for i in x:
        ans[i] = ans[i] + 1
    return ans

@jit(nopython=True)
def bin_weight_count(x, weights, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints. Also sums all weights for each value. If a value n is found at position i, count[n] += 1 and sumw[n] += weight[i].

The number of bins (of size 1) is one larger than the largest value in x. If minlength is specified, there will be at least this number of bins in the output array (though it will be longer if necessary, depending on the contents of x). Each bin gives the number of occurrences of its index value in x.

Parameters
----------
x : array_like, 1 dimension, nonnegative ints
    Input array.
    
weights : array_like
    Weights, array broadcastable with x (e.g. first dimension of weights is len(x)).
    
minlength : int
    A minimum number of bins for the output array.

Returns
----------
sumw : ndarray of the same type as weights
    The sum of the weights in each bin. The shape of out is equal to `(max(minlength, np.amax(x)+1), wheights.shape[1:])`.
    
count : 1D ndarray of ints
    The number of elements of x in each bin. The length of count is `max(minlength, np.amax(x)+1)`.

"""
    mx = x.max()
    mn = x.min()
    assert mn >= 0
    ans_size = max(mx+1, minlength)
    count = np.zeros((ans_size,), dtype=np.int64)
    sumw = np.zeros((ans_size,) + weights.shape[1:], dtype=weights.dtype)
    for p,i in enumerate(x):
        count[i] += 1
        sumw[i] += weights[p]
    return sumw, count
    
def bin_weight_countdd(ipos, shape, weights=None):
    """Count number of occurrences of each D-uple of coordinates in array of non-negative integer coordinates. Also sums all weights for each value. If coordinate number p has indices (i,j,k), count[i,j,k] += 1 and sumw[i,j,k] += weight[p].

In any dimension d, no coordinate should be larger or equal than shape[d] or lower than 0. 

Parameters
----------
ipos : array_like, shape (P, D), positive ints
    Discretized coordinates on the grid.
    
shape : tuple of int of size D
    The multidimentional shape of the output array.
    
weights : array_like, shape (P, F), optional
    Weights, array broadcastable with ipos (e.g. first dimension of weights is len(ipos)).

Returns
----------
sumw : ndarray of the same type as weights
    The sum of the weights in each bin. The shape of out is equal to `(*shape, *wheights.shape[1:])`. If weights is None sumw is not returned.
    
count : 1D ndarray of ints
    The number of elements of ipos in each bin. The length of count is `shape`.

"""
    assert ipos.shape[1] == len(shape)
    # Case of no coordinates inputed
    if len(ipos) == 0:
        count = np.zeros(shape, dtype=np.int64)
        if weights is None:
            return count
        sumw = np.zeros(shape + weights.shape[1:], dtype=weights.dtype)
        return sumw, count
    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(ipos.T, shape)
    if weights is None:
        # Bin xy and assign it to the flattened histmat.
        return bin_count(xy, minlength = np.prod(shape)).reshape(shape)
    # Bin xy (with and without weights) and assign it to the
    # flattened histmat.
    sumw, count = bin_weight_count(xy, weights, minlength = np.prod(shape))
    # Shape into a proper matrix
    count = count.reshape(shape)
    sumw = sumw.reshape(shape + weights.shape[1:])
    return sumw, count

class Grid:
    """A class to manage D-dimensional rectilinear grids"""
    def __init__(self, edges):
        """Parameters
        ----------
        bins : A sequence of arrays describing the monotonically increasing bin
              edges along each dimension."""
        for d, e in enumerate(edges):
            if np.any(e[:-1] > e[1:]):
                raise ValueError('`bins[{}]` must be monotonically increasing'.format(d))
        self.edges = [np.array(e) for e in edges]
        
    def save(self, fname):
        """Save to a file in a human readable format"""
        with open(fname, 'w') as f:
            f.write("Grid\n")
            for edge in self.edges:
                f.write(" ".join(["%g"%e for e in edge]) + "\n")
        
    @property
    def ndim(self):
        """Dimensionality of the grid"""
        return len(self.edges)
        
    @property
    def nbins(self):
        """Shape of the array used for binning, margins included"""
        return tuple(len(e)+1 for e in self.edges)
    
    @property
    def shape(self):
        """Shape of the output arrays"""
        return tuple(s-2 for s in self.nbins)
    
    def mask(self):
        """What are the useful points in the output?"""
        return np.ones(self.shape, bool)
    
    def mesh(self):
        """Obtain the coordinates of cell centers"""
        if self.ndim != 2:
            raise NotImplemented("mesh is implemented only in 2D")
        x,y = np.transpose([0.5*(e[1:]+e[:-1]) for e in self.edges])
        #transpose coordinates to be consistent with input
        Y, X = np.meshgrid(y, x)
        return np.column_stack((X.ravel(), Y.ravel()))
    
    def areas(self):
        """areas of the grid cells"""
        p = 1.
        for e in self.edges:
            p = np.multiply.outer(p, np.diff(e))
        return p.reshape(self.shape)
    
    def low_high_edges(self, dim):
        """Lowest and highest edges in dimension dim"""
        return self.edges[dim][0], self.edges[dim][-1]
    
    def digitize(self, pos):
        """snap positions to grid, assigning to it the index of the edge immediately larger to it.
            Coordinates that are below the lowest edge will have index 0.
            Coordinates that are above the highest edge will have index len(edges[d]).
        
        Parameters
        ----------
        pos : A (P,D) array of coordinates
        """
        if pos.ndim == 1:
            return np.searchsorted(self.edges[0], pos, side='right')
        return np.column_stack([
            np.searchsorted(e, x, side='right')
            for x,e in zip(pos.T, self.edges)
            ])
    
    def check_digitize(self, pos, qos=None):
        """If a second set of coordinates qos is defined, set to 0 the digitized coordinates that are not equal in all dimensions between pos and qos"""
        ipos = self.digitize(pos)
        if pos.ndim == 1:
            ipos = ipos[:,None]
        if qos is not None:
            iqos = self.digitize(qos)
            if qos.ndim == 1:
                iqos = iqos[:,None]
            #digitized coordinate must be equal in all dimensions
            ipos[np.any(ipos != iqos, axis=1)] = 0
        return ipos
    
    def count_sum_discreet(self, pos, fields, qos=None):
        """sum per grid element the values of scalar fields defined only at coordinates pos. 
        If a second set of coordinates qos is defined, bins only when both coordinates belong to the same grid element"""
        ipos = self.check_digitize(pos, qos)
        sumw, count = bin_weight_countdd(ipos, self.nbins, weights=fields)
        #trim sides where the coordinates outside of the grid were binned.
        core = self.ndim*(slice(1, -1),)
        return sumw[core], count[core]
    
    def count(self, pos, qos=None):
        """count how many points per grid element. Coordinates strictly below the lowest edge or above or equal to the highest edge in any dimension are not counted.
        If a second set of coordinates qos is defined, bins only when both coordinates belong to the same grid element"""
        ipos = self.check_digitize(pos, qos)
        count = bin_weight_countdd(ipos, self.nbins)
        #trim sides where the coordinates outside of the grid were binned.
        core = self.ndim*(slice(1, -1),)
        return count[core]
    
    def sum_discreet(self, pos, fields, qos=None):
        """sum per grid element the values of a scalar field defined only at coordinates pos.
        If a second set of coordinates qos is defined, bins only when both coordinates belong to the same grid element"""
        sumw, count = self.count_sum_discreet(pos, fields, qos)
        return sumw
    
    def mean_discreet(self, pos, fields, qos=None):
        """average per grid element the values of a scalar field defined only at coordinates pos.
        If a second set of coordinates qos is defined, bins only when both coordinates belong to the same grid element"""
        sumw, count = self.count_sum_discreet(pos, fields, qos)
        while count.ndim < sumw.ndim:
            count = count[...,None]
        return sumw/np.maximum(1, count)
        
        
class RegularGrid(Grid):
    """A class to manage D-dimensional regularly spaced rectilinear grids"""
    def __init__(self, offsets, steps, nsteps):
        """In dimension d, the lowest edge of the grid is at `offset[d]` and the highest is at `offset[d]+nsteps[d]*steps[d]`.
        
        
        Parameters
        ----------
        offsets : sequence of D coordinates 
        
        steps : sequence of D coordinates
        
        nsteps : sequence of D positive ints"""
        assert len(offsets) == len(steps)
        assert len(steps) == len(nsteps)
        for d, n in enumerate(nsteps):
            if int(n) != n or n <= 0:
                raise ValueError('`nsteps` must be positive integers'.format(d))
        self.offsets = np.array(offsets)
        self.steps = np.array(steps)
        self.nsteps = np.array(nsteps, np.int64) + 1
        
    def save(self, fname):
        """Save to a file in a human readable format"""
        with open(fname, 'w') as f:
            f.write("Regular\n")
            f.write(" ".join(["%g"%o for o in self.offsets]) + "\n")
            f.write(" ".join(["%g"%s for s in self.steps]) + "\n")
            f.write(" ".join(["%d"%n for n in self.nsteps -1]) + "\n")
        
    @property
    def ndim(self):
        """Dimensionality of the grid"""
        return len(self.offsets)
        
    @property
    def nbins(self):
        """Shape of the array used for binning, margins included"""
        return tuple(n+1 for n in self.nsteps)
    
    def mesh(self):
        """Obtain the coordinates of cell centers"""
        if self.ndim != 2:
            raise NotImplemented("mesh is implemented only in 2D")
        x,y = np.transpose(self.offsets + (0.5+np.transpose([np.arange(n) for n in self.nsteps-1])) * self.steps)
        #transpose coordinates to be consistent with input
        Y, X = np.meshgrid(y, x)
        return np.column_stack((X.ravel(), Y.ravel()))
    
    def areas(self):
        """areas of the grid cells"""
        return np.full(self.shape, np.prod(self.steps))
    
    def low_high_edges(self, dim):
        """Lowest and highest edges in dimension dim"""
        return self.offsets[dim], self.offsets[dim] + self.steps[dim] * (self.nsteps[dim]-1)
    
    def digitize(self, pos):
        """snap positions to grid, assigning to it the index of the edge immediately larger to it.
            Coordinates that are below the offset will have index 0.
            Coordinates that are above the highest edge will have index steps[d].
        
        Parameters
        ----------
        pos : A (P,D) array of coordinates
        """
        if pos.ndim == 1:
            return digitize_regular(pos, self.offsets[0], self.steps[0], self.nsteps[0])
        return np.column_stack([
            digitize_regular(x, offset, step, nstep)
            for x, offset, step, nstep in zip(pos.T, self.offsets, self.steps, self.nsteps)
            ])
    
    
    
class PolarGrid(Grid):
    """A class to manage 2-dimensional polar grids"""
    def __init__(self, radii, ncells, theta_offset=0):
        """A ploar grid is made of N concentric rings containing C equally-spaced cells.
        
        
        Parameters
        ----------
        radii : sequence of N+1 monotonically increasing radii
        
        ncells : integer or sequence of N integers
        
        theta_offset : angle of the first edge, or sequence of N angles"""
        assert len(radii) > 1
        if np.any(np.array(radii)<0):
            raise ValueError('`radii` must be positive')
        if np.any(radii[:-1] > radii[1:]):
            raise ValueError('`radii` must be monotonically increasing')
        if np.isscalar(ncells):
            ncells = np.full(len(radii)-1, ncells)
        if np.isscalar(theta_offset):
            theta_offset = np.full(len(radii)-1, theta_offset)
        assert len(radii) == len(ncells)+1
        assert len(theta_offset) == len(ncells)
        self.radii = np.array(radii)
        self.sqradii = self.radii**2
        self.ncells = np.array(ncells, np.int64)
        self.theta_offset = np.array(theta_offset)
        self.encells = np.zeros(len(ncells)+2)
        self.encells[1:-1] = self.ncells
        self.etheta_offset = np.zeros(len(theta_offset)+2)
        self.etheta_offset[1:-1] = self.theta_offset
        
    def save(self, fname):
        """Save to a file in a human readable format"""
        with open(fname, 'w') as f:
            f.write("Polar\n")
            f.write(" ".join(["%g"%o for o in self.radii]) + "\n")
            f.write(" ".join(["%d"%s for s in self.ncells]) + "\n")
            f.write(" ".join(["%g"%n for n in self.theta_offset]) + "\n")
            
    @property
    def ndim(self):
        """Dimensionality of the grid"""
        return 2
    
    @property
    def nbins(self):
        """Shape of the array used for binning, margins included"""
        return (len(self.radii)+1, self.ncells.max()+2)
    
    @property
    def shape(self):
        """Shape of the output arrays"""
        return (len(self.radii)-1, self.ncells.max())
    
    def mask(self):
        """What are the useful points in the output?"""
        m = np.zeros(self.shape, bool)
        for i, n in enumerate(self.ncells):
            m[i, :n] = True
        return m
    
    def mesh(self):
        """Obtain the coordinates of cell centers"""
        rs = np.repeat(0.5*(self.radii[1:] + self.radii[:-1]), self.shape[-1]).reshape(self.shape)
        thetas = (self.theta_offset[:,None] + 2*np.pi * (0.5+np.arange(self.ncells.max()))[None,:] / self.ncells[:,None])
        #case of a single cell that must be shown at the center
        rs[self.ncells==1] = 0
        #may need to rotate 90° to be consistent with axis orientation
        X = rs * np.cos(thetas)
        Y = rs * np.sin(thetas)
        mask = self.mask()
        return np.column_stack((X[mask], Y[mask]))
    
    def areas(self):
        """areas of the grid cells"""
        return np.repeat(np.diff(self.radii**2)/self.ncells, self.shape[-1]).reshape(self.shape)
        
    def low_high_edges(self, dim):
        """Lowest and highest edges in dimension dim"""
        return -self.radii[-1], self.radii[-1]
        
    def digitize(self, pos):
        """snap positions to grid, assigning to it the index of the edge immediately larger to it.
            Coordinates that are below the offset will have index 0.
            Coordinates that are above the highest edge will have index steps[d].
        
        Parameters
        ----------
        pos : A (P,D) array of coordinates
        """
        rsq = np.sum(pos**2, -1)
        ir = np.searchsorted(self.sqradii, rsq, side='right')
        theta = np.arctan2(pos[:,1], pos[:,0])
        itheta = digitize_regular(
            np.mod((theta - self.etheta_offset[ir])/(2*np.pi), 1) * self.encells[ir],
            0, 1, self.ncells.max()
        )
        #count theta=0 in the first cell of the anulus rather than discarding it.
        itheta[itheta==0] = 1
        #include the origin if the first annulus is a single cell that includes the origin
        if self.radii[0] == 0 and self.ncells[0] == 1:
            ir[ir==0] = 1
        return np.column_stack((ir, itheta))
        

def load(fname):
    """Load a grid from a file"""
    with open(fname) as f:
        typ = f.readline()[:-1]
        if typ == "Grid":
            edges = [map(float, line[:-1].split()) for line in f]
            return Grid(edges)
        elif typ == "Regular":
            offsets = np.array(list(map(float, f.readline()[-1].split())))
            steps = np.array(list(map(float, f.readline()[-1].split())))
            nsteps = np.array(list(map(int, f.readline()[-1].split())))
            return RegularGrid(offsets, steps, nsteps)
                
        
            