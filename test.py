import copy

a = [1, 2, 3, 4, 1]


def ab(c, b=1):
    a = copy.deepcopy(c)
    a.append(b)

    return a


print(a)
print(ab(a))
print(a)

# getattr(object, name, default=None): # known special case of getattr
