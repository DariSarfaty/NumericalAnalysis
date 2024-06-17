import Tools
import numpy as np


A = np.array([[4, 8, 4, 0],
                    [1, 4, 7, 2],
                    [1, 5, 4, -3],
                    [1, 3, 0, -2]], dtype=float)

n, m = np.shape(A)
if n !=m:
    print("matrix must be square")
    exit()

"""Creation of the identity matrix:"""
I = np.zeros_like(A)
for i in range(n):
    I[i,i] = 1

L, U = Tools.crout(A)
print("The matrices L and U:")
print(L)
print(U)


inverse = Tools.transpose(np.array(Tools.LU_decomposition(A, I)))
print("The inverse matrix:")
print(inverse)

print("The original matrix multiplied by the calculated inverse matrix:")
print(np.array(Tools.matrix_mult(A,inverse)))