import math 
from module import Module, Parameter

# datset 

points = [
    (0.1, 0.8), (0.2, 0.3), (0.3, 0.6), (0.4, 0.2),   # red
    (0.7, 0.6), (0.9, 0.2), (0.8, 0.9), (0.6, 0.4),   # green
]
labels = [0, 0, 0, 0, 1, 1, 1, 1]





# sigmoid 
def sigmoid(s):
    if s >= 0:
        return 1 / (1.0 + math.exp(-s))
    return math.exp(s) / (1.0 + math.exp(s))






# linear : a module with parms 

class Linear(Module):
    def __init__(self):
        super().__init__()
        self.w1 = Parameter(0.0)
        self.w2 = Parameter(0.0)
        self.b = Parameter(0.0)

    def forward(self, x1, x2):
        return self.w1.value * x1 + self.w2.value * x2+ self.b.value
    

class Network(Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear()

    def forward(self, x1, x2):
        return self.layer.forward(x1, x2)

# loss 
def loss(model):
    total = 0.0
    for (x1, x2), y in zip(points, labels):
        p = sigmoid(model.forward(x1, x2))
        if y == 1:
            total += -math.log(p)
        else:
            total += -math.log(1 - p)
    return total / len(points)


def accuracy(model):
    correct = 0 
    for (x1, x2), y in zip(points, labels):
        prediction = 1 if model.forward(x1, x2) > 0 else 0 
        if prediction == y:
            correct += 1
        return correct / len(points)



model = Network()

print(model.named_parameters())


# hand tuning 
model.layer.w1.value = 1.0
model.layer.w2.value = 0.0
model.layer.b.value = -0.5
print("loss:", loss(model))
print(accuracy(model))



# fit 
step = 0.05 
for i in range(2000):
    for p in model.parameters():
        best = loss(model)
        for direction in (+step, - step):
            old = p.value
            p.value = old + direction 
            if loss(model) < best:
                best = loss(model)
            else:
                p.value = old 

print("fitted", model.named_parameters())
print("fitted los :", loss(model))
print(" fitted accuracy:", accuracy(model))

    

    