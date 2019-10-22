from texture import *
from numpy.testing import *
from scipy.spatial import cKDTree as KDTree
from grid import RegularGrid

def test_1D_regular():
    grid = RegularGrid([0], [1], [3])
    pos = np.arange(-1,4, 0.1)[:,None]
    tree = KDTree(pos)
    edges = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 0.15))
        for j in sorted(js)
        if i<j
        ])
    sumw, count = bin_texture(pos, edges, grid)
    assert_array_almost_equal(sumw, np.full((3,1), 0.3))
    assert_array_equal(count, np.full((3), 30))
    assert_array_almost_equal(sumw/count[:,None], np.full((3,1), 0.01))
    
def test_1D_notregular():
    grid = RegularGrid([0], [1], [3])
    pos = np.array([0.1,0.5,0.9, 1.5, 1.9, 2.2, 2.3])[:,None]
    tree = KDTree(pos)
    edges = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 0.5))
        for j in sorted(js)
        if i<j
        ])
    sumw, count = bin_texture(pos, edges, grid)
    assert_array_equal(count, [6, 5, 7])
    assert_array_almost_equal(sumw, np.array([6*0.16, 3*0.16+1*0.09+1*0.16, 2*0.09+2*0.16+3*0.01])[:,None])
    
def test_2D_4branches():
    grid = RegularGrid([-1.2,-1.2], [0.8,0.8], [3,3])
    pos = np.array([[0,0], [-1,0], [0,-1], [0,1], [1,0]])
    tree = KDTree(pos)
    edges = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 1.5))
        for j in sorted(js)
        if i<j
        ])
    sumw, count = bin_texture(pos, edges, grid)
    assert_array_equal(count, [[1, 4, 1], [4, 4, 4], [1, 4, 1]])
    assert_array_almost_equal(sumw, [
        [[1,-1,1], [4,0,2], [1,1,1]],
        [[2,0,4], [2,0,2], [2,0,4]],
        [[1,1,1], [4,0,2],[1,-1,1]]
    ])