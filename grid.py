import numpy as np

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
