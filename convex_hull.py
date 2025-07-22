import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def get_blocking_edges(movement_vector, hull_points):
    blocking_edges = []
    for i in range(len(hull_points)):
        a = hull_points[i]
        b = hull_points[(i + 1) % len(hull_points)]
        edge = b - a

        # Compute the inward normal (points inside the polygon)
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)

        # Check if movement vector m pushes out through this edge
        if np.dot(movement_vector, normal) < 0:
            blocking_edges.append((a, b, normal))
    return blocking_edges


def perpendicular_distance_to_edge(p, a, normal):
    # Signed perpendicular distance from point p to the infinite line defined by edge (a, b)
    return np.dot(p - a, normal)

def compute_convex_hull(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

def perpendicular_movement_to_edge(movement_vector, normal):
    # Project movement vector m onto the normal vector
    proj_length = np.dot(movement_vector, normal)
    return proj_length
