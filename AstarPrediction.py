from matplotlib import pyplot as plt
import numpy as np
import os
import cv2


# path visualization
def visualize_path(grid, path, start, goal):
    rows, cols = grid.shape

    # Create a new grid for visualization
    vis_grid = np.copy(grid)

    # Plot the path
    if path:
        for i, (row, col) in enumerate(path):
            if i > 0:  # Don't color the start point
                vis_grid[row, col] = 0  # Black for path

    # Plot start and goal
    vis_grid[start] = 0  # Black for start
    vis_grid[goal] = 0  # Black for goal

    return vis_grid


# expand node to get surrounding neighbors
def get_neighbors(current, grid):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for dx, dy in directions:
        x, y = current[0] + dx, current[1] + dy
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 255:
            neighbors.append((x, y))

    return neighbors


# heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # manhattan's distance


# get obstacle distance from a cell
def get_obstacle_distance(current, grid):
    obstacle_dis = []

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            # print(x, y)
            if grid[x, y] == 128:
                obstacle_dis.append(((x, y), heuristic(current, (x, y))))
    obstacle_dis.sort(key=lambda x: x[1])  # sort by distance
    return obstacle_dis


# get dynamic cost
def get_dynamic_cost(grid, cost_map, max_cost, wall_cost):
    cmap = np.copy(cost_map)
    print("Calculating dynamic cost...")
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != 128:
                harmonic_sum = 0.0
                for d in get_obstacle_distance((x, y), grid):
                    # sum of 1 / distance to obstacles
                    harmonic_sum += (1 / d[1])
                cost = (harmonic_sum - np.min(cost_map)) / \
                    (np.max(cost_map) - np.min(cost_map))  # normalize cost
                # apply max cost to 'adjust stickiness'
                cmap[x, y] = cost * max_cost + 1
    return cmap


# normal A* search
def a_star(grid, cost_map, start, goal, max_cost, wall_cost):
    row, col = grid.shape
    frontier = []
    explored = {start: 0}
    heur = {start: heuristic(start, goal)}
    parent = {}
    frontier.append((start, heur[start]))

    while frontier:
        frontier.sort(key=lambda c: c[1])
        current, _ = frontier.pop(0)

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1]

        for n in get_neighbors(current, grid):
            new_cost = explored[current] + cost_map[n]

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
    # input image needs to be in the same directory as the script
    img = cv2.imread('5_ideal_img.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    counter = 5  # counter for file naming
    wall_cost = 999  # wall cost
    max_cost = 1.00  # constant that adjust the 'stickiness' of the path

    grid = np.copy(img)  # copy image data
    points = []
    # get start and end points from image
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0:  # points = black pixels
                points.append((x, y))
                # change from black to white to not affect the algo
                grid[x, y] = 255
    start, goal = points[0], points[1]
    print(start)
    print(goal)

    # get cost of each cell of grid
    cost_map = np.full(grid.shape, 1.0)  # default cost is 1
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 128:
                cost_map[x, y] = wall_cost
    # calculate cost of each cell, takes quite some time to load.
    cost_map = get_dynamic_cost(grid, cost_map, max_cost, wall_cost)

    path = a_star(grid, cost_map, start, goal,
                  max_cost, wall_cost)  # get solution
    out = visualize_path(grid, path, start, goal)  # visualize solution

    plt.imshow(out, cmap='gray')

    # save output image (solution)
    out_name = str(counter) + '_astar_ideal_' + str(max_cost) + '.png'
    plt.imsave(out_name, out, cmap='gray')
    print("saved output image to current working directory...")
