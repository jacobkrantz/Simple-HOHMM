
from copy import deepcopy
import random as ran

def init_matrix(rows, columns, data_type="float"):
    """
    Initialize a matrix using lists with provided size: (rows,columns)
    Args:
        rows (int)
        columns (int)
        data_type (string) must be one of:
            'float': 0.0 | 'int': 0 | 'int,int': tuple(0,0)
    Return:
        the zero matrix of specified size and type
    """
    if(data_type == 'int'):
        item = 0
    elif(data_type == 'float'):
        item = 0.0
    elif(data_type == 'int,int'):
        item = (0,0)

    matrix = []
    row = []
    for i in range(0, columns):
        row.append(item)
    for j in range(0, rows):
        matrix.append(deepcopy(row))
    return matrix

def init_3d_matrix(x, y, z, data_type="float"):
    """
    Initialize a 3-dim matrix using lists with provided size: (X,Y,Z)
    Args:
        x (int)
        y (int)
        z (int)
        data_type (string) must be one of:
            'float': 0.0 | 'int': 0 | 'int,int': tuple(0,0)
    Return:
        the zero matrix of specified size and type
    """
    if(data_type == 'int'):
        item = 0
    elif(data_type == 'float'):
        item = 0.0
    elif(data_type == 'int,int'):
        item = (0,0)

    d1 = []
    for i in range(z):
        d1.append(item)

    d2 = []
    for j in range(y):
        d2.append(deepcopy(d1))

    matrix = []
    for k in range(x):
        matrix.append(deepcopy(d2))

    return matrix

def init_matrix_uniform(row_len, column_len):
    """
    Initialize a matrix such that all rows sum to 1 and
    all elements in a row are the same.
    Args:
        row_len (int): Number of rows the matrix will have.
        column_len (int): Number of columns matrix will have.
    Returns:
        list<list<float>>: uniformly distributed matrix.
    """
    value = float(1.0 / column_len)
    row = list(map(lambda x : value, range(column_len)))
    return list(map(lambda x : deepcopy(row), range(row_len)))

def init_matrix_random(row_len, column_len):
    """
    Initialize a matrix such that all rows sum to 1 and elements are
    generated pseudo-randomly.
    Args:
        row_len (int): Number of rows the matrix will have.
        column_len (int): Number of columns matrix will have.
    Returns:
        list<list<float>>: randomly distributed matrix.
    """
    return list(map(lambda x : _make_random_row(column_len), range(row_len)))

def _make_random_row(num_elements):
    """ Generates a list of row_len random floats that sum to 1. """
    row = [ran.random() for i in range(num_elements)]
    s = sum(row)
    return [ i / s for i in row ]
