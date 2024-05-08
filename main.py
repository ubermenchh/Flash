import numpy as np 

class Array:
    def __init__(self, values):
        self.values = values 

    def __repr__(self):
        return str([f"{i}" for i in self.values])


if __name__=="__main__":
    A = np.array([2, 3, 4])
    print(A)

    B = Array([2, 3, 4])
    print(B)
