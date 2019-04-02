import numpy as np




def main():
    # Ex. A
    a1 = 13
    a2 = a3 = - 1
    N = 903
    f = 1
    A = generateSampleMatrix(a1, a2, a3, N, 1)
    b = generateSampleVector(N, f)

    #Ex. B
    resLimit = 10**-9

    roots_x = LinSolver_Jacobi(A, b, N, resLimit)


def generateSampleMatrix(a1, a2, a3, N, f):
    finalMatrix = []
    for i in range(N):
        finalMatrix.append([])
        for j in range(N):
            finalMatrix[i].append(0)

    for i in range(N):
        for j in range(N):
            if i == j:
                # middle diagonal
                finalMatrix[i][j] = a1
                # lower diagonals
                if j - 1 >= 0:
                    finalMatrix[i][j - 1] = a2
                if j - 2 >= 0:
                    finalMatrix[i][j - 2] = a3
                # upper diagonals
                if j + 1 < N:
                    finalMatrix[i][j + 1] = a2
                if j + 2 < N:
                    finalMatrix[i][j + 2] = a3
    return finalMatrix

def generateSampleVector(N, f):
    vec = [np.sin(f+1) for _ in range(N)]
    return vec


def LinSolver_Jacobi(A, b, N, resLimit):
    if(N < 1):
        print("I can't work... n < 1")
        return

    normRes = 1

    #TODO how does variety of init values in x affects no. of iterations?
    x = [1 for _ in range(N)] # init x

    while normRes > resLimit:
        x_prev = x.copy()
        for i in range(N):
            sum_1 = 0
            sum_2 = 0
            for j in range(i - 1):
                sum_1 += A[i][j] * x_prev[j]
            for j in range(i, N):
                sum_2 += A[i][j] * x_prev[j]
        x[i] = (b[i] - sum_1 - sum_2)/A[i][i]
        normRes = calcNormRes(x, x_prev)


def calcNormRes(vec_x, vec_prevX):
    normRes_squared = list(map(lambda x1, x2: (x1 - x2) ** 2, vec_x, vec_prevX))
    normRes_squared = sum(normRes_squared)
    return np.sqrt(normRes_squared)

main()
