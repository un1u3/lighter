class TensorData:
    def __init__(self, storage, shape, strides = None):
        self._storage = storage 
        self.shape = shape
        self.strides = strides or self.strides_from_shape(shape)

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
# import torch
# t = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(t.stride())    
# print(t.shape)       

def shapes_broadcast(shape1, shape2):
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = (1,) * (len2 - len1) + shape1
    elif len2 < len1:
        shape2 = (1,) * (len1 - len2) + shape2

    for a, b in zip(shape1, shape2):
        if a != b and a != 1 and b != 1:
            return False
    return True


def broadcast_shape(shape1, shape2):
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = (1,) * (len2 - len1) + shape1
    elif len2 < len1:
        shape2 = (1,) * (len1 - len2) + shape2

    result = []
    for a, b in zip(shape1, shape2):
        result.append(max(a, b))   # the bigger of the two "wins" at each position
    return tuple(result)


def broadcast_index(big_index, big_shape, small_shape):

    small_index = []
    offset = len(big_shape) - len(small_shape)   # small_shape is padded on the left conceptually
    for i, size in enumerate(small_shape):
        big_dim = i + offset
        if size == 1:
            small_index.append(0)                # stretched dimension: always read index 0
        else:
            small_index.append(big_index[big_dim])
    return tuple(small_index)



print(shapes_broadcast((2, 3), (3,)))              
print(shapes_broadcast((2, 3), (2,)))              
print(broadcast_shape((2, 3), (3,)))              
print(broadcast_index((1, 2), (2, 3), (3,)))        
print(broadcast_index((1, 2), (2, 3), (1, 3)))     

def position_to_index(position, shape, strides):
    index = []
    for stride in strides:
        index.append(position // stride)
        position = position % stride
    return tuple(index)

# tensor operations 
def tensor_map(fn):
    def _map(in_tensor):
        out = TensorData([0.0] * len(in_tensor._storage), in_tensor.shape)
        for i in range(len(in_tensor._storage)):
            index = position_to_index(i, in_tensor.shape, in_tensor.strides)
            out.set(index, fn(in_tensor.get(index)))
        return out
    return _map


def tensor_zip(fn):
    def _zip(a, b):
        out_shape = broadcast_shape(a.shape, b.shape)
        out = TensorData([0.0] * product(out_shape), out_shape)
        for i in range(product(out_shape)):
            out_index = position_to_index(i, out_shape, out.strides)
            a_index = broadcast_index(out_index, out_shape, a.shape)
            b_index = broadcast_index(out_index, out_shape, b.shape)
            out.set(out_index, fn(a.get(a_index), b.get(b_index)))
        return out
    return _zip


def tensor_reduce(fn, start):
    def _reduce(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1                     # collapse this dimension to size 1
        out = TensorData([start] * product(out_shape), tuple(out_shape))
        for i in range(len(a._storage)):
            index = list(position_to_index(i, a.shape, a.strides))
            index[dim] = 0
            out_index = tuple(index)
            current = out.get(out_index)
            out.set(out_index, fn(current, a._storage[i]))
        return out
    return _reduce


# small helpers used above
def product(shape):
    p = 1
    for s in shape:
        p *= s
    return p

def position_to_index(position, shape, strides):
    index = []
    for stride in strides:
        index.append(position // stride)
        position = position % stride
    return tuple(index)




a = TensorData([1, 2, 3, 4, 5, 6], shape=(2, 3))
negated = tensor_map(lambda x: -x)(a)
print(negated._storage)   

b = TensorData([10, 20, 30], shape=(3,))
summed = tensor_zip(lambda x, y: x + y)(a, b)
print(summed._storage)    

row_sums = tensor_reduce(lambda x, y: x + y, 0.0)(a, dim=1)
print(row_sums._storage)  



# pytorch implementation 
# import torch  
# a = torch.tensor([[1,2,3],[4,5,6]])
# print(-a)                      
# print(a + torch.tensor([10,20,30]))   
# print(a.sum(dim=1))              



class Tensor:
    def __init__(self, data, history=None):
        self.data = data            # a TensorData
        self.history = history
        self.grad = None

    def __add__(self, other):
        out_data = tensor_zip(lambda x, y: x + y)(self.data, other.data)
        result = Tensor(out_data, history=("add", self, other))
        return result

    def __mul__(self, other):
        out_data = tensor_zip(lambda x, y: x * y)(self.data, other.data)
        result = Tensor(out_data, history=("mul", self, other))
        return result

    def sum(self, dim):
        out_data = tensor_reduce(lambda x, y: x + y, 0.0)(self.data, dim)
        result = Tensor(out_data, history=("sum", self, dim))
        return result




negated = tensor_map(lambda x: -x)(a)
print(negated._storage)

summed = tensor_zip(lambda x, y: x + y)(a, b)
print(summed._storage)

row_sums = tensor_reduce(lambda x, y: x + y, 0.0)(a, dim=1)
print(row_sums._storage)