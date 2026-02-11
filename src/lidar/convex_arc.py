import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def is_convex_in_polar(radii, angles_rad):
    """
    Determines if a set of points defined in polar coordinates forms a convex shape.

    Args:
        radii (list or np.array): List of radial distances.
        angles_rad (list or np.array): List of angles in radians.

    Returns:
        bool: True if the shape formed by the points is convex, False otherwise.
    """
    if len(radii) < 3:
        # A shape needs at least 3 points to be considered convex/concave
        return True

    # 1. Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles_rad)
    y = radii * np.sin(angles_rad)
    points_cartesian = np.vstack((x, y)).T

    # 2. Compute the convex hull of the points
    # The convex hull is the smallest convex polygon that contains all points [1]
    hull = ConvexHull(points_cartesian)

    # 3. Check if ALL original points are part of the convex hull vertices
    # If the shape is already convex, every single input point must be on the hull boundary.
    # If a point is inside the hull boundary, the original shape was concave.
    hull_vertices_indices = set(hull.vertices)
    all_points_are_vertices = len(hull_vertices_indices) == len(points_cartesian)

    # Optional: Visualization (uncomment to see the plots)
    # visualize_shape_and_hull(points_cartesian, hull, all_points_are_vertices)

    return all_points_are_vertices

def visualize_shape_and_hull(points, hull, is_convex):
    """Helper function to plot the points and their convex hull."""
    plt.figure(figsize=(6, 6))
    plt.plot(points[:,0], points[:,1], 'o', label='Original Points')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=2, color='red', label='Convex Hull Edge' if simplex[0] == hull.simplices[0][0] else "")
    plt.title(f"Shape is Convex: {is_convex}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# --- Example Usage ---

# Example 1: A Convex Shape (e.g., a circle-ish shape)
# Radii are all 1, angles uniformly distributed.
angles_convex = np.linspace(0, 2 * np.pi, 20, endpoint=False)
radii_convex = np.ones(20) * 1.0

is_convex_shape = is_convex_in_polar(radii_convex, angles_convex)
print(f"Shape 1 is convex: {is_convex_shape}") # Output: True


# Example 2: A Concave Shape (e.g., a star or a 'Pac-Man' shape)
angles_concave = np.linspace(0, 2 * np.pi, 10, endpoint=False)
radii_concave = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])

is_concave_shape = is_convex_in_polar(radii_concave, angles_concave)
print(f"Shape 2 is convex: {is_concave_shape}") # Output: False

# Example 3: Another Convex Shape (a triangle)
angles_triangle = np.array([0, 2*np.pi/3, 4*np.pi/3])
radii_triangle = np.array([1.0, 1.0, 1.0])

is_triangle_convex = is_convex_in_polar(radii_triangle, angles_triangle)
print(f"Shape 3 (triangle) is convex: {is_triangle_convex}") # Output: True
