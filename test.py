import pickle
import math
import torch


def maxValue(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m = len(grid)
    n = len(grid[0])
    valueSum = [[0] * n for i in range(m)]
    for a in range(m + n - 1):
        i = 0
        j = a - i
        while j >= 0:
            if (i < m) and (j < n):
                valueSum[i][j] = grid[i][j] + max(visit(i - 1, j, valueSum), visit(i, j - 1,  valueSum))
            i += 1
            j -= 1
    return valueSum[m - 1][n - 1]


def visit(i, j, grid):
    if (i < 0) or (j < 0):
        return 0
    else:
        return grid[i][j]

grid =[[1,3,1],[1,5,1],[4,2,1]]

maxValue(grid)