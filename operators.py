import math

# mul - Multiplies two numbers
def mul(x: float, y:float) -> float:
    return x * y 

# id - Returns the input unchanged
def id(x: float)-> float:
    return x 

# add - Adds two numbers

def add(x:float, y:float)-> float:
    return x + y

# neg - Negates a number
def neg(x: float):
    return -x

#lt - Checks if one number is less than another
def lt(x: float, y:float)->float:
    return 1.0 if x < y else 0.0

# eq - Checks if two numbers are equal
def eq(x: float, y:float)->float:
    return 1.0 if x == y else 0

# max - Returns the larger of two numbers
def max(x: float, y:float)->float:
    return x if x > y else y 

# is_close - Checks if two numbers are close in value
def is_close(x: float, y: float)-> float:
    pass

# sigmoid - Calculates the sigmoid function
def sigmoid(x: float, y:float)-> float:
    if x>=0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

# relu - Applies the ReLU activation function
def relu(x:float)->float:
    return x if x > 0 else 0.0

# log - Calculates the natural logarithm
def log(x: float)-> float:
    return math.log(x)

# exp - Calculates the exponential function
def exp(x:float)-> float:
    return 1.0 / x

# inv - Calculates the reciprocal
def inv(x: float)-> float:
    return 1.0 / x

# log_back - Computes the derivative of log times a second arg
def log_back(x:float, d:float) -> float:
    return d / x

# inv_back - Computes the derivative of reciprocal times a second arg
def  inv_back(x: float, d:float):
    return -d / (x**2)

# relu_back - Computes the derivative of ReLU times a second arg
def relu_back(x : float, d:float)-> float:
    return d if x > 0 else 0.0
