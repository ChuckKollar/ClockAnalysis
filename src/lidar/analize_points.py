import numpy as np
from circle_fit import taubinSVD  #

def check_proximity(points, max_distance):
    """
    Checks if all points are close together by ensuring the maximum
    distance between any two points is within a threshold.
    """
    if len(points) < 2:
        return True  # Trivial case
    points_np = np.array(points)
    # Calculate all pairwise distances
    # This creates a matrix of distances, then finds the maximum distance
    max_dist = np.max(np.linalg.norm(points_np[:, None, :] - points_np[None, :, :], axis=-1))
    return max_dist <= max_distance


def check_collinearity(points, tolerance=0.5):
    """
    Checks if points form a straight line using the cross product method.
    If all points are collinear, the cross product will be close to zero for
    any set of three points (accounting for floating point error).
    """
    if len(points) < 3:
        return False  # Need at least 3 points to form an arc (or line)

    p1 = np.array(points[0])
    p2 = np.array(points[1])

    for i in range(2, len(points)):
        p3 = np.array(points[i])
        # Vector 1 (p1 -> p2) and Vector 2 (p1 -> p3)
        v1 = p2 - p1
        v2 = p3 - p1
        # Cross product (2D approximation: z-component)
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross_product) > tolerance:
            # print(f"Cross product {abs(cross_product)}")
            return False  # Not all points are on the same line

    return True  # All points are approximately collinear


def check_arc_properties(points, max_residual_error, min_radius_arc):
    """
    Fits a circle to the points and checks the fit quality (residual error)
    and radius to confirm it's an arc, not a line (infinite radius) or noise.
    """
    if len(points) < 3:
        return False

    coords = np.array(points)
    # taubinSVD returns x, y (center), r (radius), and sigma (residual error)
    xc, yc, r, sigma = taubinSVD(coords)

    # If the residual error is small, the points lie well on a circle.
    is_good_fit = sigma <= max_residual_error
    # A line can be seen as an arc with infinite radius; we ensure the radius is finite (not extremely large).
    is_arc_not_line = r < 1e6  # Set a large threshold for "infinite" radius

    return is_good_fit and is_arc_not_line, r, sigma


def analyze_points(points, max_proximity_dist=20.0, max_fit_error=0.9, min_arc_radius=10.0):
    """
    Determines if points are close, form an arc, and not a line.
    """
    if not check_proximity(points, max_proximity_dist):
        return False, "Points are not close together."

    if check_collinearity(points):
        return False, "Points form a line (or are collinear)."

    is_arc, radius, error = check_arc_properties(points, max_fit_error, min_arc_radius)
    if is_arc:
        return True, f"Points form an arc (Radius: {radius:.2f}, Error: {error:.4f})."
    else:
        return False, "Points do not fit an arc well enough or the radius is too large."
