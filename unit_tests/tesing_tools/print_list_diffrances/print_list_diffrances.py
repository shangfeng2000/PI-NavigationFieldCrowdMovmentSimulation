def print_2D_list_difrances(excpected_list: [[]], given_list: [[]]):
    if len(excpected_list) != len(given_list):
        print("List row(s)")
        return
    for x in range(0, len(excpected_list)):
        if len(excpected_list[x]) != len(given_list[x]):
            print("List column(s) not equal")

        for y in range(0, len(excpected_list[0])):
            if excpected_list[x][y] != given_list[x][y]:
                print("Position of miss match : {} {}".format(x, y))
                print("Excpected value {}  given value {}".format(excpected_list[x][y], given_list[x][y]))
