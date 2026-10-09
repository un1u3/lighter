import operators
def map(fn):
    def _map(numbers):
        results = []
        for n in numbers:
            results.append(fn(n))
        return results
    return _map



def zipWith(fn):
    def _zipWith(a, b):
        results = []
        for x, y in zip(a,b):
            results.append(fn(x,y))
        return results
    return _zipWith


def hello():
    def hi():
        print("hi ran")
    def hi2():
        print("hi2 ran")
    
    return hi2

print(hello()())