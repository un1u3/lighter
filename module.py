class Module:
    def __init__(self,):
        self.training = True
        self.children = {}
        self.params = {}

    def eval(self):
        self.training = False
        for child in self.children.values():
            child.eval()

    def train(self):
        self.training = True 
        for child in self.children.values():
            child.train()

    def parameters(self):
        result = []
        for parm in self.params.values():
            result.append(parm)

        for child in self.children.values():
            result.extend(child.parameters())
        return result


    def named_parameters(self):
        result = []
        for name, parm in self.params.items():
            result.append((name, parm))

        for child_name, child in self.children.items():
            for name, parm in child.named_parameters():
                result.append((child_name + "." + name, parm))
        return result
            




root = Module()
a = Module()
b = Module()
c = Module()


root.children["a"] = a
root.children["b"] = b 
b.children["c"] = c
root.params["w"] = 1.0
a.params["w"] = 2.0
b.params["w"] = 3.0
b.params["bias"] = 4.0
c.params["w"] = 5.0

root.eval()
print(root.training, a.training, b.training, c.training)   # False False False False

root.train()
print(root.training, a.training, b.training, c.training)   # \


print(root.parameters())   # [1.0, 2.0, 3.0, 4.0, 5.0]
print(b.parameters())      # [3.0, 4.0, 5.0]

print("*"*10)

print(c.named_parameters())     # [('w', 5.0)]
print(b.named_parameters())     # [('w', 3.0), ('bias', 4.0), ('c.w', 5.0)]
print(root.named_parameters())  # [('w', 1.0), ('a.w', 2.0), ('b.w', 3.0), ('b.bias', 4.0), ('b.c.w', 5.0)]



class Spy:
    def __setattr__(self, key, val):
        print("someone set", key, "=", val)
        super().__setattr__(key, val)    # do the normal save too

s = Spy()
s.x = 5
print(s.x)    # 5, works now