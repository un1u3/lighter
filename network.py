import random 
from numeric import Scalar, backpropagate
class Neuron:
    def __init__(self, n_inputs):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Scalar(0.0)

    def forward(self, xs):
        total = self.b 
        for w, x in zip(self.w, xs):
            total = total + w * x 
        return total 

    def params(self):
        return self.w + [self.b]



class Network:
    def __init__(self, hidden = 4):
        self.layer1 = [Neuron(2) for _ in range(hidden)]
        self.layer2 = Neuron(hidden)

    def forward(self, x1, x2):
        xs = [Scalar(x1), Scalar(x2)]
        h = [n.forward(xs).relu() for n in self.layer1]
        return self.layer2.forward(h) 

    def params(self):
        ps = []
        for n in self.layer1:
            ps += n.params()
        return ps + self.layer2.params()



points = [(0.2, 0.2), (0.3, 0.1), (0.8, 0.9), (0.7, 0.8),
          (0.2, 0.8), (0.1, 0.7), (0.9, 0.2), (0.8, 0.3)]
labels = [1, 1, 1, 1, 0, 0, 0, 0]

random.seed()
net = Network(hidden = 4)
lr = 0.05 


for epoch in range(500):
    total =  Scalar(0.0)
    for (x1, x2), y in zip(points, labels):
        p = net.forward(x1, x2).sigmoid()
        if y == 1:
            loss = Scalar(0.0) - p.log()
        else:
            loss = Scalar(0.0) - (Scalar(1.0)-p).log()
        total = total + loss

    for p in net.params():
        p.derivative = 0.0
    backpropagate(total)


    for p in net.params():
        p.data -= lr* p.derivative

    if epoch % 100 == 0:
        print(epoch, round(total.data, 3))

correct = sum((net.forward(x1, x2).data > 0) == (y == 1)
              for (x1, x2), y in zip(points, labels))

print("accuracy:", correct / len(points)) 


# pytorch 
import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(2, 4),   # layer1: 4 neurons
    nn.ReLU(),         # the bend
    nn.Linear(4, 1),   # layer2: combine
)
opt = torch.optim.SGD(net.parameters(), lr=0.5)
