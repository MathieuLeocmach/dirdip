from texture.grid import *
from numpy.testing import *

def test_bin_weight_count():
    x = np.array([1,1,5,5,2,0]*1000)
    weights1D = np.repeat(2**np.arange(6), 1000).astype(float)
    s,c = bin_weight_count(x, weights1D, 7)
    assert_array_equal(s, [10512., 20970., 10512.,     0.,     0., 21006., 0.])
    assert_array_equal(c, [1000, 2000, 1000,    0,    0, 2000, 0])
    
def test_digitize_regular():
    xs = np.array([-1,0,1])
    assert_array_equal(digitize_regular(xs, 0.1, 1, 3), [0,0,1])
    assert_array_equal(digitize_regular(xs, -0.1, 1, 3), [0,1,2])

def test_grid1D():
    grid = Grid([np.arange(0,10,2)])
    assert_equal(grid.ndim, 1)
    assert_array_equal(grid.edges[0], np.arange(0,10,2))
    #inputs a list of zero coordinates
    xs = np.zeros((0,1))
    assert_array_equal(grid.count(xs), [0,0,0,0])
    #inputs a list of coordinates
    xs = np.arange(-1,12)
    assert_array_equal(grid.count(xs), [2,2,2,2])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [2,2,2,2])
    assert_array_equal(grid.mean_discreet(xs, f), [1,1,1,1])
    f2 = np.ones((xs.shape[0], 2))
    assert_array_equal(grid.sum_discreet(xs, f2), np.full((4,2), 2))
    assert_array_equal(grid.mean_discreet(xs, f2), np.ones((4,2)))
    #effect of second set of coordinates
    ys = xs
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys = xs+0.1
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys = np.copy(xs)
    ys[2] += 2
    assert_array_equal(grid.count(xs, ys), [1,2,2,2])
    assert_array_equal(grid.sum_discreet(xs, f, ys), [1,2,2,2])
    
def test_grid2D():
    grid = Grid([np.arange(0,10,2), np.arange(-3,15,3)])
    assert_equal(grid.ndim, 2)
    assert_array_equal(grid.edges[0], np.arange(0,10,2))
    #inputs a list of zero coordinates
    xs = np.zeros((0,2))
    assert_array_equal(grid.count(xs), np.zeros((4,5), int))
    #inputs a list of coordinates
    xs = np.column_stack((np.arange(-1,12), np.arange(-1,12)))
    assert_array_equal(grid.count(xs), [
        [0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [
        [0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]])
    assert_array_equal(grid.mean_discreet(xs, f), [
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]])
    f2 = np.ones((xs.shape[0], 2))
    assert_array_equal(grid.sum_discreet(xs, f2), np.dstack(2*[[
        [0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]]]))
    assert_array_equal(grid.mean_discreet(xs, f2), np.dstack(2*[[
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]]]))
    #effect of second set of coordinates
    ys = np.copy(xs)
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys = xs + 0.1
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys[2] += 3
    assert_array_equal(grid.sum_discreet(xs, f, ys), [
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]])
    assert_array_equal(grid.mean_discreet(xs, f, ys), [
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]])
    
    
        
def test_regulargrid1D():
    grid = RegularGrid([0], [2], [4])
    assert_equal(grid.ndim, 1)
    #inputs a list of zero coordinates
    xs = np.zeros((0,1))
    assert_array_equal(grid.count(xs), [0,0,0,0])
    #inputs a list of coordinates
    xs = np.arange(-1,12)+0.1
    assert_array_equal(grid.count(xs), [2,2,2,2])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [2,2,2,2])
    assert_array_equal(grid.mean_discreet(xs, f), [1,1,1,1])
    f2 = np.ones((xs.shape[0], 2))
    assert_array_equal(grid.sum_discreet(xs, f2), np.full((4,2), 2))
    assert_array_equal(grid.mean_discreet(xs, f2), np.ones((4,2)))
    #effect of second set of coordinates
    ys = xs
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys = xs+0.1
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys = np.copy(xs)
    ys[2] += 2
    assert_array_equal(grid.count(xs, ys), [1,2,2,2])
    assert_array_equal(grid.sum_discreet(xs, f, ys), [1,2,2,2])
    
def test_regulargrid2D():
    grid = RegularGrid([0, -3], [2,3], [4,5])
    assert_equal(grid.ndim, 2)
    #inputs a list of zero coordinates
    xs = np.zeros((0,2))
    assert_array_equal(grid.count(xs), np.zeros((4,5), int))
    #inputs a list of coordinates
    xs = np.column_stack((np.arange(-1,12), np.arange(-1,12)))+0.1
    assert_array_equal(grid.count(xs), [
        [0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]])
    f = np.ones(xs.shape[0])
    assert_array_equal(grid.sum_discreet(xs, f), [
        [0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]])
    assert_array_equal(grid.mean_discreet(xs, f), [
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]])
    f2 = np.ones((xs.shape[0], 2))
    assert_array_equal(grid.sum_discreet(xs, f2), np.dstack(2*[[
        [0, 2, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]]]))
    assert_array_equal(grid.mean_discreet(xs, f2), np.dstack(2*[[
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]]]))
    #effect of second set of coordinates
    ys = np.copy(xs)
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys = xs + 0.1
    assert_array_equal(grid.count(xs), grid.count(xs, ys))
    assert_array_equal(grid.sum_discreet(xs, f), grid.sum_discreet(xs, f, ys))
    ys[2] += 3
    assert_array_equal(grid.sum_discreet(xs, f, ys), [
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0]])
    assert_array_equal(grid.mean_discreet(xs, f, ys), [
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]])
