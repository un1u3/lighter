# Parameter : a small box around a learnable numer 
class Parameter:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Parameter(" + str(self.value) + ")"



# moddule; hold chiild morue and params 


class Module:
    def __init__(self):
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

    def parameters(self):          # was: def parameter(self)
        result = []
        for parm in self.params.values():
            result.append(parm)
        for child in self.children.values():
            result.extend(child.parameters())
        return result
    def named_parameters(self):
        result = []
        for child_name, child in self.children.items():
            for name, parm in child.named_parameters():     
                result.append((child_name + "." + name, parm))  
        return result

    def __setattr__(self, key, val):
        if isinstance(val ,Module):
            self.children[key] = val
        elif isinstance(val, Parameter):
            self.params[key] = val 
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
      
        params = self.__dict__["params"]
        children = self.__dict__["children"]

        if key in params:
            return params[key]
        if key in children:
            return children[key]

        raise AttributeError("no attribute called " + key)