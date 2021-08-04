def prettyprint_list(l, brk=10):
    """
    Prints a list adding a newline every brk elements
    :param l: list to print
    :param brk: number of elements after which print a newline
    :return:
    """
    if type(l) is not list:
        l = list(l)
    printed = 0
    for i in range(len(l)):
        print(l[i], end=" ")
        printed += 1
        if printed % brk == 0:
            print()
            printed = 0
