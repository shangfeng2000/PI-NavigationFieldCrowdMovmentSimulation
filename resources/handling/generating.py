

def create_txt_form_direction_map(path_filename, direction_map):
    f = open(path_filename, "w+")
    for line in direction_map:
        for point in line:
            f.write(str(point))
            f.write(' ')
        f.write('\n')
    f.close()

