from texture.texture import *
from numpy.testing import *
from scipy.spatial import cKDTree as KDTree
from texture.grid import RegularGrid

def test_appeared_disappeared():
    pairs0 = np.array([[0,1], [2,10], [10,3]])
    pairs1 = np.array([[0,1], [4,5], [10,3], [8,3]])
    pairsa, pairsd, pairsc = bonds_appeared_disapeared(pairs0, pairs1)
    assert_array_equal(pairsa, np.array([[3,8], [4,5]]))
    assert_array_equal(pairsd, np.array([[2,10]]))
    assert_array_equal(pairsc, np.array([[0,1], [3,10]]))

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
    pos0 = np.array([0.1,0.5,0.9, 1.5, 1.9, 2.2, 2.3])[:,None]
    tree = KDTree(pos0)
    edges = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 0.5))
        for j in sorted(js)
        if i<j
        ])
    assert_array_equal(edges, np.array([[0,1], [1,2], [3,4], [4,5], [4,6], [5,6]]))
    sumw0, count0 = bin_texture(pos0, edges, grid)
    assert_array_equal(count0, [6, 5, 7])
    assert_array_almost_equal(sumw0, np.array([6*0.16, 3*0.16+1*0.09+1*0.16, 2*0.09+2*0.16+3*0.01])[:,None])
    #move 2 points within the same grid element
    pos1 = np.copy(pos0)
    pos1[:2] += 0.2
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count0, count1)
    assert_array_almost_equal(sumw1, np.array([3*0.16+3*0.04, 3*0.16+1*0.09+1*0.16, 2*0.09+2*0.16+3*0.01])[:,None])
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count0, count)
    assert_array_almost_equal(sumw, np.array([2*3*0.3*(0.2-0.4), 0, 0])[:,None])
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    
    #move one point to a different grid element, without straining any existing bond
    pos1 = np.copy(pos0)
    pos1[:3] += 0.2
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count1, [5,6,7])
    assert_array_almost_equal(sumw1, np.array([5*0.16, 4*0.16+1*0.09+1*0.16, 2*0.09+2*0.16+3*0.01])[:,None])
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count, [5,5,7])
    assert_array_almost_equal(sumw, np.array([0, 0, 0])[:,None])
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    ##update topology
    tree1 = KDTree(pos1)
    edges1 = np.array([
        [i,j] 
        for i, js in enumerate(tree1.query_ball_tree(tree1, 0.5))
        for j in sorted(js)
        if i<j
        ])
    assert_array_equal(edges1, np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [4,6], [5,6]]))
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges1, grid)
    assert_array_equal(countc, np.array([5,5,7]))
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.array([0,3,0]))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.array([0,3*0.16,0])[:,None])
    
def test_2D_4branches():
    grid = RegularGrid([-1.2,-1.2], [0.8,0.8], [3,3])
    pos0 = np.array([[0,0], [-1,0], [0,-1], [0,1], [1,0]], dtype=float)
    tree = KDTree(pos0)
    edges = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 1.5))
        for j in sorted(js)
        if i<j
        ])
    sumw0, count0 = bin_texture(pos0, edges, grid)
    assert_array_equal(count0, [[1, 4, 1], [4, 4, 4], [1, 4, 1]])
    assert_array_almost_equal(sumw0, [
        [[1,-1,1], [4,0,2], [1,1,1]],
        [[2,0,4], [2,0,2], [2,0,4]],
        [[1,1,1], [4,0,2],[1,-1,1]]
    ])
    #move 1 point within the same grid element
    pos1 = np.copy(pos0)
    pos1[-1] += [0.1,0]
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count0, count1)
    assert_array_almost_equal(sumw1, [
        [[1,-1,1], [4,0,2], [1,1,1]],
        [[2.21,0.1,4], [2.21,0,2], [2.21,-0.1,4]],
        [[1.21,1.1,1], [4.84,0,2],[1.21,-1.1,1]]
    ])
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count0, count)
    assert_array_almost_equal(sumw, [
        [[0,0,0], [0,0,0], [0,0,0]],
        [[0.21,0.1,0], [0.21,0,0], [0.21,-0.1,0]],
        [[0.21,0.1,0], [0.84,0,0],[0.21,-0.1,0]]
    ])
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    #flip the motif along x
    pos1 = np.array([[0,0], [1,0], [0,-1], [0,1], [-1,0]], dtype=float)
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count0, count1)
    assert_array_almost_equal(sumw0, sumw1)
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count, [[0, 0, 0], [4, 4, 4], [0, 0, 0]])
    assert_array_almost_equal(sumw, np.zeros((3,3,3)))
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    #rotate the motif by 90°
    pos1 = np.array([[0,0], [0,-1], [1,0], [-1,0], [0,1]], dtype=float)
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count0, count1)
    assert_array_almost_equal(sumw0, sumw1)
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count, [[0, 0, 0], [0, 4, 0], [0, 0, 0]])
    assert_array_almost_equal(sumw, np.zeros((3,3,3)))
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    #change topology only
    pos1 = np.copy(pos0)
    edges1 = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 1.1))
        for j in sorted(js)
        if i<j
        ] + [[1,4], [2,3]])
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges1, grid)
    assert_array_equal(countc, [[0, 2, 0], [2, 4, 2], [0, 2, 0]])
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, [[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    assert_array_equal(countd, [[1, 2, 1], [2, 0, 2], [1, 2, 1]])
    assert_array_almost_equal(sumT, [
        [[-1,1,-1], [2,0,-2], [-1,-1,-1]],
        [[-2,0,2], [4,0,4], [-2,0,2]],
        [[-1,-1,-1], [2,0,-2],[-1,1,-1]]
    ])
    
def test_2D_3branches():
    grid = RegularGrid([-1.2,-1.2], [0.8,0.8], [3,3])
    pos0 = np.array([[0,0], [-1,0], [0,-1], [0,1]], dtype=float)
    tree = KDTree(pos0)
    edges = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 1.5))
        for j in sorted(js)
        if i<j
        ])
    sumw0, count0 = bin_texture(pos0, edges, grid)
    assert_array_equal(count0, [[1, 4, 1], [3, 3, 3], [0, 0, 0]])
    assert_array_almost_equal(sumw0, [
        [[1,-1,1], [4,0,2], [1,1,1]],
        [[1,-1,3], [1,0,2], [1,1,3]],
        [[0,0,0], [0,0,0],[0,0,0]]
    ])
    #move 1 point within the same grid element
    pos1 = np.copy(pos0)
    pos1[-1] += [0,0.1]
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count0, count1)
    assert_array_almost_equal(sumw1, [
        [[1,-1,1], [4,0.1,2.21], [1,1.1,1.21]],
        [[1,-1,3], [1,0,2.21], [1,1.1,3.63]],
        [[0,0,0], [0,0,0],[0,0,0]]
    ])
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count0, count)
    assert_array_almost_equal(sumw, sumw1-sumw0)
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    #flip the motif along x
    pos1 = np.array([[0,0], [1,0], [0,-1], [0,1]], dtype=float)
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count0, [[1, 4, 1], [3, 3, 3], [0, 0, 0]])
    assert_array_almost_equal(sumw1, [
        [[0,0,0], [0,0,0],[0,0,0]],
        [[1,1,3], [1,0,2], [1,-1,3]],
        [[1,1,1], [4,0,2], [1,-1,1]]
    ])
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count, [[0, 0, 0], [3, 3, 3], [0, 0, 0]])
    assert_array_almost_equal(sumw, [
        [[0,0,0], [0,0,0],[0,0,0]],
        [[0,2,0], [0,0,0], [0,-2,0]],
        [[0,0,0], [0,0,0],[0,0,0]]
    ])
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    #rotate the motif by 90°
    pos1 = np.array([[0,0], [0,-1], [1,0], [-1,0]], dtype=float)
    sumw1, count1 = bin_texture(pos1, edges, grid)
    assert_array_equal(count1, count0.T)
    assert_array_almost_equal(sumw1, [
        [[1,-1,1], [3,-1,1],[0,0,0]],
        [[2,0,4], [2,0,1], [0,0,0]],
        [[1,1,1], [3,1,1], [0,0,0]]
    ])
    sumw, count = bin_geometrical_changes(pos0, pos1, edges, grid)
    assert_array_equal(count, [[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    assert_array_almost_equal(sumw, [
        [[0,0,0], [0,0,0],[0,0,0]],
        [[0,0,0], [1,0,-1], [0,0,0]],
        [[0,0,0], [0,0,0],[0,0,0]]
    ])
    ## there should be no topology change
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges, grid)
    assert_array_equal(count, countc)
    assert_array_almost_equal(sumw, sumB)
    assert_array_equal(counta, np.zeros_like(count))
    assert_array_equal(countd, np.zeros_like(count))
    assert_array_almost_equal(sumT, np.zeros_like(sumw))
    #change topology only
    pos1 = np.copy(pos0)
    edges1 = np.array([
        [i,j] 
        for i, js in enumerate(tree.query_ball_tree(tree, 1.1))
        for j in sorted(js)
        if i<j
        ] + [[2,3]])
    sumB, countc, sumT, counta, countd = bin_changes(pos0, pos1, edges, edges1, grid)
    assert_array_equal(countc, [[0, 2, 0], [2, 3, 2], [0, 0, 0]])
    assert_array_almost_equal(sumB, np.zeros_like(sumw))
    assert_array_equal(counta, [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert_array_equal(countd, [[1, 2, 1], [1, 0, 1], [0, 0, 0]])
    assert_array_almost_equal(sumT, [
        [[-1,1,-1], [-2,0,-2], [-1,-1,-1]],
        [[-1,1,3], [0,0,4], [-1,-1,3]],
        [[0,0,0], [0,0,0],[0,0,0]]
    ])
