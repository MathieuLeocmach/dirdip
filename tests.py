from grid import *
from numpy.testing import *

def test_grid():
    grid = Grid([np.arange(0,10,2)])
    assert_equal(grid.dim, 1)
    xs = np.arange(-1,12)[:,None]
    assert_array_equal(grid.bins[0], np.arange(0,10,2))
    assert_array_equal(grid.count(xs), [2,2,2,3])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [2,2,2,3])
    assert_array_equal(grid.mean_discreet(xs, f), [1,1,1,1])
