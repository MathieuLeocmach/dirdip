import numpy as np
from numba import jit

#implementation using loops and numba
@jit(nopython=True)
def bin_weight_count(x, weights, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints. Also sums all weights for each value. If a value n is found at position i, count[n] += 1 and out[n] += weight[i].

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
out : ndarray of the same type as weights
    The sum of the weights in each bin. The shape of out is equal to `(max(minlength, np.amax(x)+1), wheights.shape[1:])`.
    
count : 1D ndarray of ints
    The number of elements of x in each bin. The length of count is `max(minlength, np.amax(x)+1)`.

"""
    mx = x.max()
    mn = x.min()
    assert mn >= 0
    ans_size = max(mx+1, minlength)
    n = np.zeros((ans_size,), dtype=np.int64)
    s = np.zeros((ans_size,) + weights.shape[1:], dtype=weights.dtype)
    for i,w in zip(x, weights):
        n[i] += 1
        s[i] += w
    return s,n
    
def bin_weight_countdd(ipos, weights, shape):
    """Count number of occurrences of each D-uple of coordinates in array of non-negative integer coordinates. Also sums all weights for each value. If coordinate number p has indices (i,j,k), count[i,j,k] += 1 and out[i,j,k] += weight[p].

In any dimension d, no coordinate should be larger or equal than shape[d] or lower than 0. 

Parameters
----------
ipos : array_like, shape (P, D), nonnegative ints

    Input array.
    
weights : array_like, shape (P, F)

    Weights, array broadcastable with ipos (e.g. first dimension of weights is len(ipos)).
    
shape : sequence of int of size D

    The multidimentional shape of the output array.

Returns
----------
out : ndarray of the same type as weights
    The sum of the weights in each bin. The shape of out is equal to `(*shape, *wheights.shape[1:])`.
    
count : 1D ndarray of ints
    The number of elements of ipos in each bin. The length of count is `shape`.

"""
    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(ipos, shape)
    # Bin xy (with and without weights) and assign it to the
    # flattened histmat.
    s,c = np.bincount(xy, weights, minlength = np.prod(shape))
    # Shape into a proper matrix
    c = c.reshape(shape)
    s = s.reshape(shape + c.shape[1:])
    return s,n

class Grid:
    """A class to manage D-dimensional rectilinear grids"""
    def __init__(self, bins):
        """Parameters
        ----------
        bins : A sequence of arrays describing the monotonically increasing bin
              edges along each dimension."""
        self.bins = bins
        
    @property
    def dim(self):
        """Dimensionality of the grid"""
        return len(self.bins)
    
    def digitize(self, pos):
        """snap positions to grid. 
            Coordinates that are below the lowest bin will have index 0.
            Coordinates that are above the highest bin will have index len(bin).
        
        Parameters
        ----------
        pos : A (P,D) array of coordinates
        """
        if pos.dim == 1:
            return np.digitize(pos, self.bins[0])
        return np.column_stack([np.digitize(x, b) for x,b in zip(pos.T, self.bins)])
    
    def count(self, pos):
        """count how many points per grid element. Coordinates strictly below the lowest bin or strictly above the highest bin in any dimension are not counted."""
        hist, bins = np.histogramdd(pos, self.bins)
        return hist
    
    def sum_discreet(self, pos, field):
        """sum per grid element the values of a scalar field defined only at coordinates pos"""
        hist, bins = np.histogramdd(pos, self.bins, weights=field)
        return hist
    
    def mean_discreet(self, pos, field):
        """average per grid element the values of a scalar field defined only at coordinates pos"""
        s = self.sum_discreet(pos, field)
        n = self.count(pos)
        return s/np.maximum(1, n)
