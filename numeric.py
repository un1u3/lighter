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

    def log(self):
        result = Scalar(log_forward(self.data))
        result.history = (log_backward, self)
        return result

    def exp(self):
        result = Scalar(exp_forward(self.data))
        result.history = (exp_backward, self)
        return result

    def inv(self):
        result = Scalar(inv_forward(self.data))
        result.history = (inv_backward, self)
        return result

    def relu(self):
        result = Scalar(relu_forward(self.data))
        result.history = (relu_backward, self)
        return result

    def sigmoid(self):
        result = Scalar(sigmoid_forward(self.data))
        result.history = (sigmoid_backward, self)
        return result

    def __sub__(self, other):
        return self + Scalar(-1.0) * other



def add_backward(a, b, d_output):
    return d_output, d_output 

def mul_backward(a, b, d_output):
    return d_output *b,  d_output  * a 

a = Scalar(2.0)
b = Scalar(3.0)
c = a + b
d = c * a

d_grad = 1.0

backward_fn, left, right = d.history
c_grad_from_d, a_grad_from_d = backward_fn(left.data, right.data, d_grad)
print("c_grad_from_d:", c_grad_from_d)
print("a_grad_from_d:", a_grad_from_d)


grads = {}

def add_grad(scalar, amount):
    key = id(scalar)
    if key not in grads:
        grads[key] = 0.0
    grads[key] += amount


def collect(scalar, seen= None):
    if seen is None:
        seen = []
    if scalar in seen:
        return seen 

    seen.append(scalar)

    if scalar.history is not None:
        backward_fn, left, right = scalar.history
        collect(left, seen)
        collect(right, seen)
    return seen 

def topological_sort(scalar, seen=None, order=None):
    if seen is None:
        seen = []
        order = []

    if scalar in seen:
        return order

    seen.append(scalar)

    if scalar.history is not None:
        if len(scalar.history) == 2:
            backward_fn, inp = scalar.history
            topological_sort(inp, seen, order)
        else:
            backward_fn, left, right = scalar.history
            topological_sort(left, seen, order)
            topological_sort(right, seen, order)

    order.insert(0, scalar)
    return order


def backpropagate(variable, deriv=1.0):
    order = topological_sort(variable)
    grads = {id(variable): deriv}

    for scalar in order:
        d_output = grads.get(id(scalar), 0.0)

        if scalar.history is None:
            scalar.derivative = d_output
        elif len(scalar.history) == 2:
            backward_fn, inp = scalar.history
            (contrib,) = backward_fn(inp.data, d_output)
            grads[id(inp)] = grads.get(id(inp), 0.0) + contrib
        else:
            backward_fn, left, right = scalar.history
            left_contrib, right_contrib = backward_fn(left.data, right.data, d_output)
            grads[id(left)] = grads.get(id(left), 0.0) + left_contrib
            grads[id(right)] = grads.get(id(right), 0.0) + right_contrib

import math

def log_forward(a):
    return math.log(a)

def log_backward(a, d_output):
    return (d_output / a,)


def exp_forward(a):
    return math.exp(a)

def exp_backward(a, d_output):
    out = math.exp(a)
    return (d_output * out,)


def inv_forward(a):
    return 1.0 / a

def inv_backward(a, d_output):
    return (-d_output / (a ** 2),)


def relu_forward(a):
    return a if a > 0 else 0.0

def relu_backward(a, d_output):
    return (d_output if a > 0 else 0.0,)


def sigmoid_forward(a):
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    return math.exp(a) / (1.0 + math.exp(a))

def sigmoid_backward(a, d_output):
    out = sigmoid_forward(a)
    return (d_output * out * (1.0 - out),)


a = Scalar(2.0)
b = Scalar(3.0)
c = a + b
d = c * a

backpropagate(d)
print(a.derivative, b.derivative)   




def train(model, points, labels, lr=0.1, epochs=500):
    for epoch in range(epochs):
        total_loss = Scalar(0.0)

        for (x1, x2), y in zip(points, labels):
            out = model.forward(Scalar(x1), Scalar(x2))
            p = out.sigmoid()

            if y == 1:
                point_loss = Scalar(0.0) - p.log()
            else:
                one_minus_p = Scalar(1.0) - p
                point_loss = Scalar(0.0) - one_minus_p.log()

            total_loss = total_loss + point_loss

        backpropagate(total_loss)

        for param in model.parameters():
            param.value.data -= lr * param.value.derivative
            param.value.derivative = None

        if epoch % 50 == 0:
            print(epoch, total_loss.data)




a = Scalar(2.0)
out = a.sigmoid()
backpropagate(out)
print(a.derivative)