
def reduction(matrix):
    """ takes a nXn+1 matrix and solves it"""
    (rows, cols) = (len(matrix), len(matrix[0]))
    """reduce down:"""
    for pivot in range(rows):
        a = matrix[pivot][pivot]
        copy = matrix[pivot].copy()
        matrix[pivot] = [x/a for x in copy]
        for row in range(pivot + 1, rows):
            b = matrix[row][pivot]
            matrix[row] = [elem - piv*b for elem, piv in zip(matrix[row],matrix[pivot])]
    """reduce up:"""
    for pivot in range(rows - 1, -1, -1):
        for row in range(pivot - 1, -1, -1):
            c = matrix[row][pivot]
            matrix[row] = [elem - piv * c for elem, piv in zip(matrix[row], matrix[pivot])]
    return matrix

