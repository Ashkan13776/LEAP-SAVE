import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree


def generate_uniform_points_in_sphere(num_points,surface_bias=5):
    phi = np.random.uniform(0, np.pi * 2, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(costheta)
    r = u ** (1./surface_bias)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.vstack((x, y, z)).T


def calculate_distance_from_center(pos):
    return np.sqrt(np.sum(pos**2))

def create_graph_with_features(num_points, k, surface_bias=2):
    positions = generate_uniform_points_in_sphere(num_points, surface_bias)
    G = nx.Graph()

    for i, pos in enumerate(positions):
        distance_from_center = calculate_distance_from_center(pos)
        G.add_node(i, pos=pos, distance_from_center=distance_from_center)

    tree = KDTree(positions)
    for i in range(num_points):
        distances, indices = tree.query(positions[i], k+1)
        for index in indices[1:]:
            G.add_edge(i, index)

    return G, positions