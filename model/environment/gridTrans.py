def word_to_grid(min_x,max_x,min_y,max_y,row_num,col_num,word_x,word_y):
    grid_width = (max_x - min_x) / col_num
    grid_height = (max_y - min_y) / row_num
    # Calculate column index
    col_index = int((word_x - min_x) / grid_width)
    # Calculate row index
    row_index = int((max_y - word_y) / grid_height)
    grid_x = col_index
    grid_y = row_index
    return (grid_x,grid_y)

def grid_to_word(min_x,max_x,min_y,max_y,row_num,col_num,grid_x,grid_y):
    grid_width = (max_x - min_x) / col_num
    grid_height = (max_y - min_y) / row_num
    # Calculate world x coordinate
    word_x = min_x + grid_x * grid_width
    # Calculate world y coordinate, inverted due to Y increasing upwards
    word_y = max_y - grid_y * grid_height
    return (word_x,word_y)