from grid import *
from numpy.testing import *

def test_grid1D():
    grid = Grid([np.arange(0,10,2)])
    assert_equal(grid.dim, 1)
    xs = np.arange(-1,12)
    assert_array_equal(grid.bins[0], np.arange(0,10,2))
    assert_array_equal(grid.count(xs), [2,2,2,3])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [2,2,2,3])
    assert_array_equal(grid.mean_discreet(xs, f), [1,1,1,1])
    
def test_grid2D():
    grid = Grid([np.arange(0,10,2), np.arange(-3,12,3)])
    assert_equal(grid.dim, 2)
    xs = np.column_stack((np.arange(-1,12), np.arange(-1,12)))
    assert_array_equal(grid.bins[0], np.arange(0,10,2))
    assert_array_equal(grid.count(xs), [
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3]])
    assert_array_equal(grid.mean_discreet(xs, f), [
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
