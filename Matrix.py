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
        self.rows, self.cols = self.cols, self.rows

    def generatePermutationMat(self):
        I = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            I.data[i][(i + 1) % self.cols] = 1
        return I

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.mulMatrices(other)
        if isinstance(other, Vector):
            return self.mulMatVec(other)
        if isinstance(other, (int, float, complex)):
            return self.mulScalarMat(other)
        raise ("Not defined multiplication!")

    def __sub__(self, other):
        if other.rows != self.rows and other.cols != self.cols:
            raise ("Dimension doesn't match!")
        C = Matrix.fromValues(self.data)
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.data[i][j] - other[i][j]
                C.put(i, j, val)
        return C

    def mulMatrices(self, other):
        # A(self) is NxM; B(other) is MxP
        n = len(self.data)
        m_A = len(self.data[0])
        p = len(other.data[0])
        m_B = len(other.data)
        if m_A != m_B:
            raise ("Dimensions don't match!")
        m = m_A

        C = Matrix(n, p)

        for i in range(n):
            for j in range(p):
                sum = 0
                for k in range(m):
                    sum += self.data[i][k] * other.data[k][j]
                C.put(i, j, sum)
        return C

    def mulMatVec(self, other):
        if self.cols != other.size:
            raise ("Dimensions don't match!")
        newVec = Vector(other.size)
        for x in range(self.rows):
            sum = 0
            for y in range(other.size):
                sum += self.data[x][y] * other.data[y]
            newVec.put(x, sum)
        return newVec

    def mulScalarMat(self, other):
        for cell in self.data:
            cell *= other

    def print(self):
        for row in range(self.rows):
            for col in range(self.cols):
                print('{:03.2f}'.format(self.data[row][col]), end='')
                print(" ", end='')
            print()


from Vector import *
