import numpy as np
from operator import sub

def main():
    # Ex. A
    a1 = 13
    a2 = a3 = - 1
    N = 12
    f = 1
    A = generateSampleMatrix(a1, a2, a3, N, 1)
    b = generateSampleVector(N, f)

    #Ex. B
    resLimit = 10**-9

    #roots_x = LinSolver_Jacobi(A, b, N, resLimit)
    #x = np.linalg.solve(A, b)

    #print(roots_x)
    #print(x)
    tmp = []
    tmp.append(b)

    print(multiplyMatrices(A, tmp, False, True))
    print(np.matmul(A, b))



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
    x = [2 for _ in range(N)] # init x

    noIter = 0
    while normRes > resLimit:
        noIter += 1
        x_prev = x.copy()
        for i in range(N):
            sum_1 = 0
            sum_2 = 0
            for j in range(i - 1):
                sum_1 += A[i][j] * x_prev[j]
            for j in range(i, N):
                sum_2 += A[i][j] * x_prev[j]
            x[i] = (b[i] - sum_1 - sum_2)/A[i][i]
        res = calcRes(A, x, b)
        normRes = calcNorm(res, 2)
        print(normRes)

    return x


def transposeMat(M):
    n = len(M)
    m = len(M[0])

    newM = []
    for i in range(m):
        newM.append([])
        for j in range(n):
            newM[i].append(0)

    for i in range(n):
        for j in range(m):
            newM[j][i] = M[i][j]
    return newM


def multiplyMatrices(A, B, toTransposeA, toTransposeB):
    #it is not multiusable function, it's only for lists as they are
    C = []
    #A is NxM; B is MxP

    if toTransposeA:
        A = transposeMat(A)
    if toTransposeB:
        B = transposeMat(B)
    n = len(A)
    m1 = len(A[0])
    p = len(B[0])
    m = len(B)

    if m1 != m:
        print("It can't work, sizes doesnt match")
        return

    for i in range(n):
        C.append([])

    for i in range(n):
        for j in range(p):
            sum = 0
            for k in range(m):
                sum += A[i][k] * B[k][j]
            C[i].append(sum)
    return C

    # # it is not multiusable function, it's only for lists as they are
    # C = []
    # # A is NxM; B is MxP
    # n = len(A)
    # p = len(B[0])
    # m = len(B)
    #
    # for i in range(n):
    #     C.append([])
    #
    # for i in range(n):
    #     for j in range(p):
    #         sum = 0
    #         for k in range(m):
    #             sum += A[i][k] * B[k][j]
    #         C[i].append(sum)
    # return C

def calcRes(A, x, b):
    # res = y - b; y = A*x
    if isinstance(x[0],list) == False: #so it's vector
        tmpArr = []
        tmpArr.append(x)
        y = multiplyMatrices(A, tmpArr)
    else:
        y = multiplyMatrices(A, x)
    res = list(map(sub, y[0], b))
    return res

def calcNorm(res, p):
    normRes_squared = list(map(lambda x1: x1**p, res))
    normRes_squared = sum(normRes_squared)
    return np.power(normRes_squared, 1/p)




main()
