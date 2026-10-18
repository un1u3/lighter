class TensorData:
    def __init__(self, storage, shape, strides = None):
        self._storage = storage 
        self.shape = shape
        self.strides = strides or self.strides_from_shape(shape)

    def strides_from_shape(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i+ 1] * shape[i+ 1]
        return tuple(strides)

    @staticmethod
    def strides_from_shape(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
    
    def index_to_position(self, index):
        postion = 0 
        for ind, stride in zip(index, self.strides):
            postion += ind * stride
            return postion 

    def get(self, index):
        return self._storage[self.index_to_position(index)]

    def set(self, index, value):
        self._storage[self.index_to_position(index)] = value 



# the 2x3 grid: [[1,2,3],[4,5,6]]
t = TensorData([1, 2, 3, 4, 5, 6], shape=(2, 3))
print(t.strides)          
print(t.get((1, 2)))      


# pytorch implemeentation 
import torch
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t.stride())    
print(t.shape)       