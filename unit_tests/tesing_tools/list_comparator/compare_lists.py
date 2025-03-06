def compare_lists(a, b):
    for x in range(0, len(a)):
        for y in range(0, len(a[1])):

            if (a[x][y]) != (b[x][y]):
                return False
    return True