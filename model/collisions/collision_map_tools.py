# Mark fileds on collision_map by adding to them given value
# Ignores marking out of map boundries
import model.navigator.navigator as nav


def mark_location(location: (int, int), collision_map, marking_value: int):
    (location_y, location_x) = location
    if location_x < 0 or location_x >= len(collision_map[0]):
        return False
    if location_y < 0 or location_y >= len(collision_map):
        return False
    collision_map[location_y][location_x] = collision_map[location_y][location_x] + marking_value
    return True


def mark_circle_obstacle(center_location: (int, int), radius: int, collision_map: [[int]], marking_value: int):
    (loc_y, loc_x) = center_location
    for x in range(loc_x - radius, loc_x + radius+1):
        for y in range(loc_y - radius, loc_y + radius+1):
            distance = nav.get_distance_beteween_points(center_location, (y, x))
            if distance <= radius:
                mark_location((y, x), collision_map, marking_value)


def mark_squere_obstacle(center_location: (int, int), border_lenght: int, collision_map: [[int]], marking_value: int):
    (loc_y, loc_x) = center_location
    size = round(border_lenght / 2)

    for x in range(loc_y - size, loc_y + size + 1):
        for y in range(loc_x - size, loc_x + size + 1):
            mark_location((y, x), collision_map, marking_value)
