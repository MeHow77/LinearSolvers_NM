import numpy as np
import time
from operator import sub, mul
from Matrix import *
from Vector import *


def generateSampleMatrix(a1, a2, a3, N):
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
    vec = [np.sin(i * (f + 1)) for i in range(N)]
    return vec


def main():
    # Ex. A
    a1 = 13
    a2 = a3 = - 1
    N = 903
    f = 1
    A = generateSampleMatrix(a1, a2, a3, N)
    m_A = Matrix.fromValues(A)
    b = generateSampleVector(N, f)
    m_b = Vector.fromValues(b)

    # Ex. B
    resLimit = 10 ** -9
    # start = time.time()
    # solJacobi = IterLinSolver(m_A, m_b, N, resLimit, Jacobi)
    # solJacobi[0].print()
    # print(solJacobi[1])
    # end = time.time()
    # print(end - start)
    #
    # start = time.time()
    # solGS = IterLinSolver(m_A, m_b, N, resLimit, GaussSiedel)
    # solGS[0].print()
    # print(solGS[1])
    # end = time.time()
    # print(end - start)

    # TODO compare time of above functions

    # Ex. C
    a1 = 3
    a2 = a3 = -1

    # A = generateSampleMatrix(a1, a2, a3, N)
    # m_A = Matrix.fromValues(A)
    # b = generateSampleVector(N, f)
    # m_b = Vector.fromValues(b)
    # solJacobi_C = LinSolver(m_A, m_b, N, resLimit, Jacobi)
    # print(solJacobi_C[1])
    # solJacobi_C[0].print()
    #
    # solGS_C = LinSolver(m_A, m_b, N, resLimit, GaussSiedel)
    # print(solGS_C[1])
    # solGS_C[0].print()

    # They don't converge;

    # Ex. D
    val = [[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]]
    A = Matrix.fromValues(val)
    LinSolver(A, m_b, 4, resLimit)

def IterLinSolver(m_A, vec_b, N, resLimit, method):
    if (N < 1):
        raise ("Size of vector is negative!")

    normRes = 1
    # TODO how does variety of init values in x affects no. of iterations?
    soln = Vector(N)
    noIter = 0

    while normRes > resLimit:
        noIter += 1
        soln.data = method(m_A.data, vec_b.data, N, soln.data)
        res = calcRes(m_A, soln, vec_b)
        normRes = calcNorm(res, 2)
    return soln, noIter


def LinSolver(m_A, vec_b, N, resLimit):
    L, U = CalcLUMatrices(m_A, N)
    y = Vector(N)
    y = U*vec_b


def CalcLUMatrices(m_A, N):
    # L = I, U = A
    L = Matrix(N, N)

    for i in range(N):
        L.put(i, i, 1)
    U = Matrix.fromValues(m_A.data)

    for k in range(0, N - 1):
        for j in range(k + 1, N):
            valueL = U.data[j][k] / U.data[k][k]
            L.put(j, k, valueL)

            rowValues_prev = list(map(lambda x: x * valueL, U.data[k][k:]))
            rowValues = [a - b for a, b in zip(U.data[j][k:], rowValues_prev)]
            U.data[j][k:] = rowValues
    return L, U

def GaussSiedel(m_A, vec_b, N, result):
    for i in range(N):
        sum = 0
        for j in range(N):
            if j != i:
                sum += m_A[i][j] * result[j]
        result[i] = (vec_b[i] - sum) / m_A[i][i]
    return result


def Jacobi(m_A, vec_b, N, result):
    prev_soln = result.copy()

    for i in range(N):
        sum = 0
        for j in range(N):
            if j != i:
                sum += m_A[i][j] * prev_soln[j]
        result[i] = (vec_b[i] - sum) / m_A[i][i]
    return result


#

def calcRes(A, x, b):
    y = A * x
    res = y - b
    return res


def calcNorm(v_res, p):
    normRes_squared = list(map(lambda x1: x1 ** p, v_res.data))
    normRes_squared = sum(normRes_squared)
    return np.power(normRes_squared, 1 / p)


def matSub(A, B):
    C = []
    for i in range(len(A)):
        C.append(A[i] - B[i])
    return C


main()
