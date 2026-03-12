import random
from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay, distance
from toponetx.classes.simplicial_complex import SimplicialComplex
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from topomodelx.nn.simplicial.scone import SCoNe

# %load_ext autoreload
# %autoreload 2
#%%
def generate_complex(N: int = 100) -> tuple[SimplicialComplex, np.ndarray]:
    """
    Generate a simplicial complex of dimension 2 as follows:
        1. Uniformly sample N random points form the unit square and build the Delaunay triangulation.
        2. Delete triangles contained in some pre-defined disks.
    """
    points = np.random.uniform(0, 1, size=(N, 2))

    # Sort points by the sum of their coordinates
    c = np.sum(points, axis=1)
    order = np.argsort(c)
    points = points[order]

    tri = Delaunay(points)  # Create Delaunay triangulation

    # Remove triangles having centroid inside the disks
    disk_centers = np.array([[0.3, 0.7], [0.7, 0.3]])
    disk_radius = 0.15
    simplices = []
    indices_included = set()
    for simplex in tri.simplices:
        center = np.mean(points[simplex], axis=0)
        if ~np.any(distance.cdist([center], disk_centers) <= disk_radius, axis=1):
            # Centroid is not contained in some disk, so include it.
            simplices.append(simplex)
            indices_included |= set(simplex)

    # Re-index vertices before constructing the simplicial complex
    idx_dict = {i: j for j, i in enumerate(indices_included)}
    for i in range(len(simplices)):
        for j in range(3):
            simplices[i][j] = idx_dict[simplices[i][j]]

    sc = SimplicialComplex(simplices)
    coords = points[list(indices_included)]
    return sc, coords
#%%
def plot_complex(sc: SimplicialComplex, coords: np.ndarray) -> None:
    """
    Given a simplicial complex of dimension 1 or 2, plot the simplices in the plane using the coordinates of the 0-simplices in coords.
    """
    # Plot triangles
    for idx in sc.skeleton(2):
        pts = np.array([coords[idx[0]], coords[idx[1]], coords[idx[2]]])
        poly = plt.Polygon(pts, color="green", alpha=0.25)
        plt.gca().add_patch(poly)

    # Plot edges
    start = coords[np.array(sc.skeleton(1))[:, 0]]
    end = coords[np.array(sc.skeleton(1))[:, 1]]
    plt.plot(
        np.vstack([start[:, 0], end[:, 0]]),
        np.vstack([start[:, 1], end[:, 1]]),
        color="black",
        alpha=0.5,
    )

    # Plot points
    plt.scatter(coords[:, 0], coords[:, 1], color="black", s=30)
#%%
def normalize_laplacian(L):
    eigenvalues, _ = np.linalg.eigh(L)
    min_eig = eigenvalues.min()
    max_eig = eigenvalues.max()
    
    # Normalize the Laplacian such that the eigenvalues lie in [0, 1]
    L_norm = (L - min_eig * np.eye(L.shape[0])) / (max_eig - min_eig)
    
    return L_norm
#%%
def laplacian_to_incidence(L):
    num_vertices = L.shape[0]
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if L[i, j] != 0 or L[j, i] != 0:
                edges.append((i, j))
                
    num_edges = len(edges)
    B = np.zeros((num_edges, num_vertices))
    
    for idx, (i, j) in enumerate(edges):
        B[idx, i] = 1
        B[idx, j] = -1
    
    return B
