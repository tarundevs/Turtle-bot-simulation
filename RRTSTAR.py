import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Helper functions
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def steer(from_node, to_node, max_extend_length=float('inf')):
    dist = euclidean_distance(from_node, to_node)
    if dist <= max_extend_length:
        return to_node
    else:
        direction = (np.array(to_node) - np.array(from_node)) / dist
        new_point = np.array(from_node) + direction * max_extend_length
        return tuple(new_point.astype(int))

def collision_free(node, new_node, image):
    line_points = bresenham_line(node, new_node)
    for point in line_points:
        if image[point[1], point[0]] == 0:  # obstacle
            return False
    return True

def bresenham_line(start, end):
    """Bresenham's Line Algorithm"""
    points = []
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

class RRTStar:
    def __init__(self, start, goal, image, max_iter=1000, max_extend_length=50, goal_sample_rate=0.2):
        self.start = start
        self.goal = goal
        self.image = image
        self.max_iter = max_iter
        self.max_extend_length = max_extend_length
        self.goal_sample_rate = goal_sample_rate
        self.tree = [start]
        self.parents = {start: None}
        self.cost = {start: 0}

    def planning(self):
        for i in range(self.max_iter):
            rnd_point = self.goal if np.random.rand() < self.goal_sample_rate else (np.random.randint(0, self.image.shape[1]), np.random.randint(0, self.image.shape[0]))
            nearest_node = self.get_nearest_node(rnd_point)
            new_node = steer(nearest_node, rnd_point, self.max_extend_length)
            if not collision_free(nearest_node, new_node, self.image):
                continue
            neighbors = self.get_neighbors(new_node)
            self.tree.append(new_node)
            self.parents[new_node] = nearest_node
            self.cost[new_node] = self.cost[nearest_node] + euclidean_distance(nearest_node, new_node)
            self.rewire(new_node, neighbors)
            if euclidean_distance(new_node, self.goal) <= self.max_extend_length:
                if collision_free(new_node, self.goal, self.image):
                    self.tree.append(self.goal)
                    self.parents[self.goal] = new_node
                    self.cost[self.goal] = self.cost[new_node] + euclidean_distance(new_node, self.goal)
                    print(f"Goal reached in iteration {i}")
                    break
        return self.get_final_path()

    def get_nearest_node(self, rnd_point):
        tree_kdtree = cKDTree(self.tree)
        _, idx = tree_kdtree.query(rnd_point)
        return self.tree[idx]

    def get_neighbors(self, new_node, radius=50):
        tree_kdtree = cKDTree(self.tree)
        indices = tree_kdtree.query_ball_point(new_node, radius)
        return [self.tree[i] for i in indices]

    def rewire(self, new_node, neighbors):
        for neighbor in neighbors:
            new_cost = self.cost[new_node] + euclidean_distance(new_node, neighbor)
            if new_cost < self.cost[neighbor]:
                if collision_free(new_node, neighbor, self.image):
                    self.parents[neighbor] = new_node
                    self.cost[neighbor] = new_cost

    def get_final_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = self.parents.get(node)  # Use get() to avoid KeyError
        path.reverse()
        return path

# Define start and goal points
start = (133, 283)
goal = (577, 635)

# Load a black and white image
image_path = r"\\wsl.localhost\Ubuntu-22.04\home\tarunwarrier\ros_ws\ERC-hackathon-2024\hackathon_automation\src\roads.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded correctly
if image is None:
    raise ValueError(f"Could not load image from path: {image_path}")

# Run RRT*
rrt_star = RRTStar(start, goal, image)
path = rrt_star.planning()

# Display the image with the RRT* tree
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for node in rrt_star.tree:
    if rrt_star.parents[node] is not None:
        cv2.line(image_color, node, rrt_star.parents[node], (255, 0, 0), 4)  # Red color


# Draw the path
for i in range(len(path) - 1):
    cv2.line(image_color, path[i], path[i + 1], (0, 0, 255), 2)

# Show the image
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.show()
