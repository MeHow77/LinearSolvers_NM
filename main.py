import numpy as np

class Matrix:

    def __init__(self, n, m):
        self.rows = n
        self.cols = m
        self.data = []

        for i in range(n):
            self.data.append([])
            for j in range(m):
                self.data[i].append(0)

    @staticmethod
    def fromValues(values):
        obj = Matrix(len(values), len(values[0]))
        for i in range(obj.rows):
            for j in range(obj.cols):
                obj.data[i][j] = values[i][j]
        return obj


    def put(self, x, y, val):
        self.data[x][y] = val

    def transpose(self):
        n = self.rows
        m = self.cols

        newM = []
        for i in range(m):
            newM.append([])
            for j in range(n):
                newM[i].append(0)

        for i in range(n):
            for j in range(m):
                newM[j][i] = self.data[i][j]
        self.data = newM

    def __mul__(self, other):
        if isinstance(self, other) is not True:
            raise("Arguments are not matrices!")


        # A(self) is NxM; B(other) is MxP
        n = len(self.data)
        m_A = len(self.data[0])
        p = len(other.data[0])
        m_B = len(other.data)
        if m_A != m_B:
            raise("Dimensions don't match!")
        m = m_A

        C = Matrix(n,p)


        for i in range(n):
            for j in range(p):
                sum = 0
                for k in range(m):
                    sum += self.data[i][k] * other.data[k][j]
                C.put(i, j, sum)
        return C

    def print(self):
        for row in range(self.rows):
            print("|", end='')
            for col in range(self.cols):
                print(self.data[row][col], end='')
                print(" ", end='')
            print("|")




def main():
    # Ex. A
    a1 = 13
    a2 = a3 = - 1
    N = 903
    f = 1
    A = generateSampleMatrix(a1, a2, a3, N)
    m_A = Matrix.fromValues(A)
    b = generateSampleVector(N, f)
    m_B = Matrix.fromValues(b)
    m_B.print()

    #Ex. B
    resLimit = 10**-9
    # solJacobi = LinSolver_Jacobi(A, b, N, resLimit)
    # print(solJacobi[0], solJacobi[1])
    #
    # solGS = LinSolver_GaussSiedel(A, b, N, resLimit)
    # print(solGS[0], solGS[1])
    #TODO compare time of above functions

    a1 = 3
    a2 = a3 = -1

    # A = generateSampleMatrix(a1, a2, a3, N)
    # solJacobi_C = LinSolver_Jacobi(A, b, N, resLimit)
    # print(solJacobi_C[0], solJacobi_C[1])
    #
    # solGS_C = LinSolver_GaussSiedel(A, b, N, resLimit)
    # print(solGS_C[0], solGS_C[1])

    #They don't converge;




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
    vec = [np.sin (i * (f+1)) for i in range(N)]
    matrix_1xN = []
    matrix_1xN.append(vec)
    return matrix_1xN


def LinSolver_Jacobi(A, b, N, resLimit):
    if(N < 1):
        print("I can't work... n < 1")
        return

    normRes = 1
    #TODO how does variety of init values in x affects no. of iterations?
    x = [0 for _ in range(N)] # init x

    noIter = 0
    while normRes > resLimit:
        noIter += 1
        x_prev = x.copy()

        for i in range(N):
            sum = 0
            for j in range(N):
                if j != i:
                    sum += A[i][j]*x_prev[j]
            x[i] = (b[i] - sum)/A[i][i]

        res = calcRes(A, x, b)
        normRes = calcNorm(res, 2)
        #print(normRes)

    return x, noIter

def LinSolver_GaussSiedel(A, b, N, resLimit):
    if(N < 1):
        print("I can't work... n < 1")
        return

    normRes = 1
    #TODO how does variety of init values in x affects no. of iterations?
    x = [0 for _ in range(N)] # init x

    noIter = 0
    while normRes > resLimit:
        noIter += 1
        x_prev = x.copy()

        for i in range(N):
            sum = 0
            for j in range(N):
                if j != i:
                    sum += A[i][j] * x[j]
            x[i] = (b[i] - sum) / A[i][i]
        res = calcRes(A, x, b)
        normRes = calcNorm(res, 2)
        #print(normRes)

    return x, noIter


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


def calcRes(A, x, b):
    # res = y - b; y = A*x
    if isinstance(x[0],list) == False: #so it's vector
        tmpArr = []
        tmpArr.append(x)
        y = multiplyMatrices(A, tmpArr, False, True)
    else:
        y = multiplyMatrices(A, x, False, True)
    y = transposeMat(y)
    #print(np.matmul(A,x))
    res = matSub(y[0], b)
    return res

def calcNorm(res, p):
    normRes_squared = list(map(lambda x1: x1**p, res))
    normRes_squared = sum(normRes_squared)
    return np.power(normRes_squared, 1/p)

def matSub(A,B):
    C = []
    for i in range(len(A)):
        C.append(A[i] - B[i])
    return C



main()
