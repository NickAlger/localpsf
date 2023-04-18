def assert_equal(a, b):
    if not (a == b):
        raise RuntimeError(str(a) + ' =/= ' + str(b))

def assert_lt(a, b):
    if not (a < b):
        raise RuntimeError(str(a) + ' not < ' + str(b))

def assert_le(a, b):
    if not (a <= b):
        raise RuntimeError(str(a) + ' not <= ' + str(b))

def assert_gt(a, b):
    if not (a > b):
        raise RuntimeError(str(a) + ' not > ' + str(b))

def assert_ge(a, b):
    if not (a >= b):
        raise RuntimeError(str(a) + ' not >= ' + str(b))