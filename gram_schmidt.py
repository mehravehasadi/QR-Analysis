import numpy as np

def gram_schmidt_qr(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

rowsA = int(input("Enter the number of rows of matrix A: "))
colsA = int(input("Enter the number of columns of matrix A: "))

A = np.zeros((rowsA, colsA), dtype=np.float64)  # Initialize a NumPy array

print("Enter matrix A elements:")
for i in range(rowsA):
    for j in range(colsA):
        A[i, j] = float(input(f"element [{i + 1}][{j + 1}]: "))

# Perform Gram-Schmidt QR decomposition
Q, R = gram_schmidt_qr(A)

print("Matrix Q:")
print(Q)
print("\nMatrix R:")
print(R)