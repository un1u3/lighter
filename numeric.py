class Scalar:
    def __init__(self, value, history = None):
        self.data = value
        self.history = history # to store which was formed by which 

    def __add__(self, other):
        result = Scalar(self.data + other.data)
        result.history = (add_backward, self, other)
        return  result

    def __mul__(self, other):
        result = Scalar(self.data * other.data)
        result.history = (mul_backward, self, other)
        return result

    def __repr__(self):
        return "Scaler(" + str(self.data) + ")"



def add_backward(a, b, d_output):
    return d_output, d_output 

def mul_backward(a, b, d_output):
    return d_output *  d_output  * a 

a = Scalar(2.0)
b = Scalar(3.0)
c = a * b

backward_fn, left, right = c.history
grads = backward_fn(left.data, right.data, 1.0)
print(grads)   # expect (3.0, 2.0) — d(a*b)/da = b = 3.0, d(a*b)/db = a = 2.0