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

def reduce(fn, start):
    def _reduce(nums):
        total = start
        for n in nums:
            total =   fn(total, n)
        return total 
    return _reduce



def neglist(ls):
    return map(operators.neg)(ls)


def addList(ls1, ls2):
    return zipWith(operators.add)(ls1, ls2)

def sum(ls):
    return reduce(operators.add, 0.0)(ls)

def prod(ls):
    return reduce(operators.mul, 1.0)(ls)
