import numpy as np

def find_proximal_radial_points(points, d_max):
    """
    Finds consecutive points in a radial system (r, theta)
    that are within distance d_max.
    Points: List of (radius, angle_radians)
    """
    # 1. Sort points by angle (theta)
    sorted_pts = sorted(points, key=lambda x: x[1])

    # 2. Convert Polar to Cartesian for distance calculation
    # x = r * cos(theta), y = r * sin(theta)
    cartesian_pts = []
    for r, theta in sorted_pts:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        cartesian_pts.append((x, y))

    proximal_pairs = []

    # 3. Check distances between consecutive neighbors
    for i in range(len(cartesian_pts) - 1):
        p1 = np.array(cartesian_pts[i])
        p2 = np.array(cartesian_pts[i + 1])

        # Euclidean distance
        distance = np.linalg.norm(p1 - p2)

        if distance <= d_max:
            proximal_pairs.append((sorted_pts[i], sorted_pts[i + 1], distance))

    return proximal_pairs


# --- Example Usage ---
if __name__ == "__main__":
    # Points: (radius, theta_in_radians)
    data_points = [(10, 0.1), (10.2, 0.2), (5, 1.5), (10.1, 0.15), (5.1, 1.6)]
    max_distance = 2.0

    pairs = find_proximal_radial_points(data_points, max_distance)

    print(f"Pairs within distance {max_distance}:")
    for p1, p2, dist in pairs:
        print(f"Point A(r:{p1[0]}, θ:{p1[1]:.2f}) & "
              f"Point B(r:{p2[0]}, θ:{p2[1]:.2f}), Distance: {dist:.2f}")
