import numpy as np
import math
"""
To find sequential points that fall on an arc from a list of points when the arc properties
are unknown, you can use an algorithm that iteratively checks groups of points for arc-like
properties. A practical approach involves using a moving window of three points to determine
a potential arc's properties (center and radius) and then verifying if subsequent points fit
this arc within a tolerance.
 
This problem is well-suited to the Ramer-Douglas-Peucker (RDP) algorithm for curve simplification,
which identifies significant points along a curve and can help segment an arc from other shapes,
or a custom approach using the properties of circles. 
"""

def _polar_to_cartesian(r, theta):
    """
    Converts single polar coordinates (radius r, angle theta in radians)
    to Cartesian coordinates (x, y).
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

def _lidar_readings_to_cartesian(readings):
    """
    Convert the lidar readings to Cartesian coordinates.
    :param readings: [(quality, angle, distance), ...]
    :return:
    """
    return [_polar_to_cartesian(x[2], math.radians(x[1])) for x in readings]

def _get_circle_center_radius(p1, p2, p3):
    """
    Calculates the center and radius of a circle given three points.
    If the points are collinear, returns None for center and radius.
    """
    # Convert to numpy arrays for easier math
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Check if points are collinear by comparing slopes or using a cross product in 2D
    # A simple check: if cross product of vectors (p2-p1) and (p3-p2) is near zero
    # In 2D, this is checked using the determinant of the matrix formed by the vectors
    if abs((p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])) < 1e-9:
        return None, None

    # Midpoints of segments p1-p2 and p2-p3
    mid1 = (p1 + p2) / 2.0
    mid2 = (p2 + p3) / 2.0

    # Slopes of segments p1-p2 and p2-p3
    try:
        slope1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        perp_slope1 = -1 / slope1
    except ZeroDivisionError:
        perp_slope1 = None  # Vertical line, use horizontal perpendicular bisector

    try:
        slope2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
        perp_slope2 = -1 / slope2
    except ZeroDivisionError:
        perp_slope2 = None  # Vertical line, use horizontal perpendicular bisector

    # Calculate center (intersection of perpendicular bisectors)
    if perp_slope1 is None:
        center_x = mid1[0]
        center_y = perp_slope2 * (center_x - mid2[0]) + mid2[1]
    elif perp_slope2 is None:
        center_x = mid2[0]
        center_y = perp_slope1 * (center_x - mid1[0]) + mid1[1]
    else:
        center_x = (mid2[1] - mid1[1] + perp_slope1 * mid1[0] - perp_slope2 * mid2[0]) / (perp_slope1 - perp_slope2)
        center_y = perp_slope1 * (center_x - mid1[0]) + mid1[1]

    center = np.array([center_x, center_y])
    radius = np.linalg.norm(p1 - center)
    return center, radius


def find_arcs(readings, tolerance=0.1, min_arc_points=6):
    """
    Finds subsets of sequential points that form an arc.
    """
    points = _lidar_readings_to_cartesian(readings)
    arcs = []
    if len(points) < 3:
        return arcs

    # Iterate through all possible starting points of a potential arc
    i=0
    while i < len(points) - 2:
        p1 = points[i]
        p2 = points[i+1]
        p3 = points[i+2]

        center, radius = _get_circle_center_radius(p1, p2, p3)

        if center is not None:
            current_arc_indexes = (i, i+2)
            current_arc_points = [p1, p2, p3]
            # Check subsequent points to see if they fit the same arc
            if i+3 >= len(points):
                break
            k_save = 0
            for k in range(i+3, len(points)):
                k_save = k
                p_next = points[k]
                distance_to_center = np.linalg.norm(p_next - center)
                if abs(distance_to_center - radius) < tolerance:
                    current_arc_indexes = (i, k)
                    current_arc_points.append(p_next)
                else:
                    # Arc segment ends
                    i += 1
                    break

            # If enough points were found, store the arc
            if (current_arc_indexes[1] - current_arc_indexes[0]) >= min_arc_points:
                # Avoid adding overlapping or redundant arc segments if possible (basic check)
                # For simplicity, this example just collects all found segments
                arcs.append(current_arc_indexes)
                # Skip points already covered by this arc to find only maximal arcs
                i = k_save - 1
        else:
            i += 1

    return arcs


# Example Usage (quality, angle, distance):
scan = [(15, 9.921875, 374.75), (15, 8.640625, 371.25), (15, 7.90625, 368.5), (15, 6.921875, 365.5), (15, 5.5625, 363.0), (15, 3.9375, 357.25), (15, 3.078125, 357.25), (15, 1.84375, 356.0), (15, 0.703125, 355.0), (15, 359.71875, 352.75), (15, 358.71875, 352.0), (15, 357.828125, 355.0), (15, 356.796875, 359.0), (15, 355.828125, 358.25), (15, 354.828125, 356.25), (15, 353.875, 357.0), (15, 352.734375, 357.25), (15, 351.546875, 358.0), (15, 350.6875, 358.25), (15, 349.765625, 360.25), (15, 348.859375, 362.75), (15, 348.03125, 365.25), (15, 347.078125, 367.0), (15, 345.921875, 370.25), (15, 344.890625, 372.5), (15, 343.96875, 375.5), (15, 342.953125, 380.0), (15, 340.1875, 946.75), (15, 339.078125, 944.5), (15, 338.078125, 949.5), (15, 337.109375, 955.0), (15, 336.109375, 959.5), (15, 335.140625, 977.5), (13, 333.265625, 639.25), (15, 332.25, 610.5), (11, 331.171875, 590.25), (12, 329.921875, 568.0), (15, 328.859375, 547.75), (15, 327.890625, 529.25), (15, 326.78125, 512.75), (15, 325.734375, 497.5), (15, 324.296875, 482.0), (15, 323.375, 468.5), (15, 322.203125, 456.75), (15, 321.0, 444.0), (15, 320.21875, 434.0), (15, 319.015625, 423.5), (15, 317.78125, 413.75), (15, 316.890625, 404.75), (15, 315.75, 396.5), (15, 314.8125, 388.25), (15, 313.609375, 381.0), (15, 312.375, 373.75), (15, 311.28125, 367.0), (15, 310.28125, 361.0), (15, 308.84375, 355.25), (15, 307.859375, 349.25), (15, 306.65625, 343.75), (15, 305.5625, 339.0), (15, 304.453125, 334.25), (15, 303.59375, 329.25), (15, 302.8125, 325.25), (15, 301.4375, 321.25), (15, 300.453125, 317.5), (15, 299.421875, 313.5), (15, 298.421875, 310.0), (15, 297.5, 307.0), (15, 296.375, 303.5), (15, 295.28125, 300.75), (15, 294.15625, 297.75), (15, 293.078125, 295.5), (15, 291.453125, 293.25), (15, 290.359375, 290.75), (15, 289.9375, 288.75), (15, 288.34375, 287.0), (15, 287.84375, 287.75), (15, 286.25, 307.75), (15, 280.828125, 776.75), (15, 279.890625, 787.0), (8, 271.34375, 699.5), (15, 270.234375, 685.75), (15, 269.203125, 677.25), (15, 268.21875, 683.5), (12, 267.359375, 690.25), (15, 265.375, 757.75), (15, 264.375, 751.75), (15, 263.40625, 751.75), (11, 262.453125, 768.0), (4, 258.765625, 999.5), (12, 241.328125, 847.0), (14, 240.3125, 851.75), (8, 239.296875, 875.75), (7, 237.171875, 827.75), (15, 236.140625, 823.5), (15, 235.09375, 825.75), (15, 233.34375, 924.25), (15, 232.21875, 904.75), (15, 231.140625, 882.25), (15, 230.171875, 863.25), (15, 229.125, 847.0), (15, 227.921875, 833.75), (15, 227.0, 848.25), (15, 225.96875, 862.5), (15, 225.0, 875.5), (15, 224.0, 897.75), (15, 223.015625, 915.5), (15, 222.125, 930.0), (15, 221.109375, 948.25), (15, 220.09375, 968.25), (15, 219.109375, 988.25), (8, 208.09375, 620.5), (8, 207.046875, 631.25), (14, 206.140625, 632.0), (7, 205.0, 633.25), (6, 204.125, 644.25), (15, 201.109375, 647.0), (15, 196.1875, 716.75), (7, 195.203125, 724.0), (15, 193.125, 732.25), (15, 192.078125, 740.75), (15, 158.25, 161.75), (6, 96.78125, 998.0), (7, 94.625, 848.0), (7, 90.703125, 993.25), (4, 89.71875, 997.25), (14, 71.03125, 309.0), (12, 69.109375, 287.75), (15, 68.09375, 267.0), (15, 65.09375, 248.5), (15, 65.09375, 242.0), (15, 64.546875, 244.5), (15, 63.375, 246.0), (15, 62.640625, 248.0), (15, 61.984375, 250.0), (15, 61.0625, 252.0), (15, 60.15625, 254.75), (15, 58.109375, 256.75), (15, 58.0625, 259.5), (15, 57.03125, 261.75), (15, 55.25, 264.5), (15, 55.0625, 267.25), (15, 54.359375, 270.25), (15, 53.15625, 273.25), (15, 51.6875, 276.25), (15, 51.03125, 279.75), (15, 50.328125, 283.5), (15, 48.65625, 287.0), (15, 48.296875, 291.0), (15, 47.640625, 295.75), (15, 46.203125, 299.25), (15, 45.203125, 304.0), (15, 44.140625, 308.5), (15, 43.5625, 313.75), (15, 42.390625, 319.0), (15, 41.734375, 324.25), (15, 41.046875, 329.75), (15, 39.9375, 335.5), (15, 38.984375, 341.5), (15, 38.0, 348.5), (15, 37.078125, 355.5), (15, 36.359375, 363.0), (15, 35.171875, 371.0), (15, 34.484375, 378.75), (15, 33.578125, 387.5), (15, 32.453125, 396.5), (15, 31.53125, 406.5), (15, 30.46875, 416.75), (15, 29.90625, 427.5), (15, 28.8125, 438.75), (15, 27.703125, 452.0), (15, 27.15625, 464.25), (14, 25.921875, 479.5), (15, 25.296875, 494.5), (12, 24.359375, 510.5), (12, 23.453125, 529.5), (12, 22.515625, 547.75), (9, 21.5, 569.75), (9, 20.515625, 592.25), (7, 19.6875, 615.5), (15, 13.25, 917.0)]

# Find arc segments
found_arcs = find_arcs(scan, tolerance=2.0)  # Tolerance is important

print(f"Found {len(found_arcs)} potential arc segments.")
for i, arc in enumerate(found_arcs):
    print(f"Arc {i + 1} with {arc[1]-arc[0]} points: {arc}")
    # print(arc) # Uncomment to see the points
