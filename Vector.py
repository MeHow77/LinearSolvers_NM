class Vector:

    def __init__(self, n):
        self.size = n
        self.data = []

        for i in range(n):
            self.data.append(0)

    @staticmethod
    def fromValues(values):
        obj = Vector(len(values))
        for i in range(obj.size):
            obj.data[i] = values[i]
        return obj

    def put(self, i, val):
        self.data[i] = val

    def print(self, n):
        for elem in self.data:
            print(f'%0{n}.{n-1}f'%(elem), end='')
            print(" ", end='')
            print()

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.mulVecMat(other)
        # if isinstance(other, Vector):
        #     return self.mulVecMat(other)
        # if isinstance(other, (int, float, complex)):
        #     return self.mulScalarMat(other)
        raise ("Not defined multiplication!")

    def mulVecMat(self, other):
        if self.size != other.cols:
            raise ("Dimensions don't match!")
        newVec = Vector(self.size)
        for x in range(other.rows):
            sum = 0
            for y in range(self.size):
                sum += other.data[x][y] * self.data[y]
            newVec.put(x, sum)
        return newVec

    def mulScalarVec(self, num):
        for cell in self.data:
            cell *= num

    def __sub__(self, other):
        if isinstance(other, self.__class__) is not True:
            raise ("Input was not two vectos!")
        newVec = Vector(self.size)
        for i in range(self.size):
            result = self.data[i] - other.data[i]
            newVec.put(i, result)
        return newVec



from Matrix import *
