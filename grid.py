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
        self.edges = edges
        
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
        #trim sides where the coordinates outside of the grid were binned
        core = self.ndim*(slice(1, -1),)
        return sumw[core], count[core]
    
    def count(self, pos, qos=None):
        """count how many points per grid element. Coordinates strictly below the lowest edge or above or equal to the highest edge in any dimension are not counted.
        If a second set of coordinates qos is defined, bins only when both coordinates belong to the same grid element"""
        ipos = self.check_digitize(pos, qos)
        count = bin_weight_countdd(ipos, self.nbins)
        #trim sides where the coordinates outside of the grid were binned
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
        
    @property
    def ndim(self):
        """Dimensionality of the grid"""
        return len(self.offsets)
        
    @property
    def nbins(self):
        """Shape of the array used for binning, margins included"""
        return tuple(n+1 for n in self.nsteps)
    
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

