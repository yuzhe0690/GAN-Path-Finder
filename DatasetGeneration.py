from matplotlib import pyplot as plt
import numpy as np


# random obstacle generation
def generate_random_obstacle(grid_size=64, obstacle_num=1, obstacle_scale=0.2, indent=5):
    min_gap = 7
    grid = np.full((grid_size, grid_size), 255, dtype=np.uint8)
    obstacle_amt = 0
    total_attempts = 0
    max_attempts = 1000

    while obstacle_amt < obstacle_num and total_attempts < max_attempts:
        obstacle_size = np.random.randint(
            (grid_size * obstacle_scale) // 2, round(grid_size * obstacle_scale) + 1)
        x = np.random.randint(indent, grid_size - indent)
        y = np.random.randint(indent, grid_size - indent)
        shape = np.random.choice(['rectangle', 'square', 'circle'])
        print(shape, x, y, obstacle_size)
        success = False

        if shape == 'rectangle':
            rec_type = np.random.choice(np.arange(0, 2))

            if rec_type == 0:  # vertical rectangle
                scale = float(1 + np.random.randint(0, 10) / 10)
                height = round(obstacle_size * scale)
                if is_space_free(grid, x, y, obstacle_size, height, min_gap):
                    for dx in range(obstacle_size):
                        for dy in range(height):
                            if is_in_bounds(grid_size, x, y, dx, dy, shape):
                                grid[x + dx][y + dy] = 255
                    success = True
            else:  # horizontal rectangle
                scale = float(1 + np.random.randint(0, 10) / 10)
                width = round(obstacle_size * scale)
                if is_space_free(grid, x, y, width, obstacle_size, min_gap):
                    for dx in range(width):
                        for dy in range(obstacle_size):
                            if is_in_bounds(grid_size, x, y, dx, dy, shape):
                                grid[x + dx][y + dy] = 255
                    success = True
        elif shape == 'square':
            if is_space_free(grid, x, y, obstacle_size, obstacle_size, min_gap):
                for dx in range(obstacle_size):
                    for dy in range(obstacle_size):
                        if is_in_bounds(grid_size, x, y, dx, dy, shape):
                            grid[x + dx][y + dy] = 255
                success = True
        elif shape == 'circle':
            radius = obstacle_size // 2
            if is_space_free(grid, x, y, radius * 2, radius * 2, min_gap):
                for dx in range(radius):
                    for dy in range(radius):
                        if dx**2 + dy**2 < radius**2 and is_in_bounds(grid_size, x, y, dx, dy, shape):
                            grid[x + dx][y + dy] = 255
                            grid[x - dx][y - dy] = 255
                            grid[x + dx][y - dy] = 255
                            grid[x - dx][y + dy] = 255
                success = True

        if success:
            obstacle_amt += 1
            total_attempts = 0
        else:
            total_attempts += 1

        if total_attempts >= max_attempts:
            print(
                f"Warning: Could only place {obstacle_amt} obstacles out of {obstacle_num} requested")
            break

    return grid


# generate random coords
def generate_random_coords(grid, indent, excluded_quadrant=None):
    available_quadrants = [1, 2, 3, 4]
    if excluded_quadrant is not None:
        available_quadrants.remove(excluded_quadrant)

    while (True):
        quadrant = np.random.choice(available_quadrants)

        if quadrant == 1:  # botttom left
            rand_x = np.random.randint(indent, (row // 2) - 1)
            rand_y = np.random.randint(col // 2, col - 1)
        elif quadrant == 2:  # bottom right
            rand_x = np.random.randint(row // 2, row - 1)
            rand_y = np.random.randint(col // 2, col - 1)
        elif quadrant == 3:  # top left
            rand_x = np.random.randint(indent, (row // 2) - 1)
            rand_y = np.random.randint(indent, (col // 2) - 1)
        else:  # top right
            rand_x = np.random.randint(row // 2, row - 1)
            rand_y = np.random.randint(indent, (col // 2) - 1)

        if (grid[rand_x, rand_y] == 255):
            return (rand_x, rand_y), quadrant


# generate start and goal points
def generate_start_goal(grid, indent):
    start, quadrant = generate_random_coords(
        grid, indent, excluded_quadrant=None)
    goal, _ = generate_random_coords(grid, indent, quadrant)
    return start, goal


# expand node to get surrounding neighbors
def get_neighbors(current, grid):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for dx, dy in directions:
        x, y = current[0] + dx, current[1] + dy
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 255:
            neighbors.append((x, y))

    return neighbors


# Manhattans distance heuristic function
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
    # return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    # return max(b[0] - a[0], b[1] - a[1])


# path visualization
def visualize_path(grid, path, start, goal):
    # plt.figure()
    rows, cols = grid.shape

    # Create a new grid for visualization
    vis_grid = np.copy(grid)

    # Plot the path
    if path:
        for i, (row, col) in enumerate(path):
            if i > 0:  # Don't color the start point
                vis_grid[row, col] = 1  # Black for path

    # Plot start and goal
    vis_grid[start] = 1  # Black for start
    vis_grid[goal] = 1  # Black for goal

    return vis_grid


def visualize_pathless(grid, start, goal):
    rows, cols = grid.shape
    g = np.copy(grid)
    g[start] = 0
    g[goal] = 0
    return g

# check if selected coordinates to generate obstacles is free


def is_space_free(grid, x, y, shape_width, shape_height, min_gap):
    x_start = max(0, x - min_gap)
    y_start = max(0, y - min_gap)
    grid_size = len(grid[0])
    x_end = min(grid_size, x + shape_width + min_gap)
    y_end = min(grid_size, y + shape_height + min_gap)

    # Check if all the space including the gap is free (255)
    return np.all(grid[x_start:x_end, y_start:y_end] == 255)


# check if coordinates to generate the coordinate is within bounds
def is_in_bounds(grid_size, x, y, dx, dy, shape):
    if shape == 'rectangle' or shape == 'square':
        return x + dx < grid_size and y + dy < grid_size
    else:
        return x + dx < grid_size and y + dy < grid_size and x - dx >= 0 and y - dy >= 0


# calculate turning to discourage turns in the algorithm
def get_turn_cost(curr_parent, current, neighbor):
    if curr_parent is None:
        return 0
    curr_dir = (current[0] - curr_parent[0], current[1] - curr_parent[1])
    new_dir = (neighbor[0] - current[0], neighbor[1] - neighbor[1])
    return 5 if (curr_dir != new_dir) else 0


# normal A* search
def a_star(grid, start, goal):
    row, col = grid.shape
    frontier = []
    explored = {start: 0}
    heur = {start: heuristic(start, goal)}
    parent = {}
    frontier.append((start, heur[start]))

    while frontier:
        frontier.sort(key=lambda c: c[1])
        print(frontier[0])
        current, _ = frontier.pop(0)

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1]

        for n in get_neighbors(current, grid):
            curr_parent = parent.get(current)
            new_cost = explored[current] + 1 + \
                get_turn_cost(curr_parent, current, n)

            if n not in explored or explored[n] > new_cost:
                parent[n] = current
                explored[n] = new_cost
                new_heur = heuristic(n, goal) + new_cost

                if n not in [f[0] for f in frontier]:
                    frontier.append((n, new_heur))
                else:
                    for i, (node, _) in enumerate(frontier):
                        frontier[i] = (n, new_heur)
                        break

    return None


# main method
if __name__ == '__main__':
    ind = 5
    amount = 1
    counter = 6

    for i in range(amount):
        generated_map = generate_random_obstacle(
            grid_size=64, obstacle_num=6, obstacle_scale=0.20, indent=ind)
        (row, col) = generated_map.shape

        start, goal = generate_start_goal(generated_map, ind)
        print(start, goal)
        path = a_star(generated_map, start, goal)
        gt = visualize_path(generated_map, path, start, goal)
        pathless = visualize_pathless(generated_map, start, goal)
        gen_map_fname = (str(counter) + "_img.png")  # pathless map
        # gen_map_gt_fname = (str(counter) + "_log.png")  # ground truth
        plt.imsave(gen_map_fname, pathless, cmap='gray')
        # plt.imsave(gen_map_gt_fname, gt, cmap='gray')
        counter += 1
