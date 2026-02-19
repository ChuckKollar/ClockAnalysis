import numpy as np
from typing import List, Tuple
from lidar.const import lidar_readings_to_cartesian, SCAN_RADIUS_MM
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set the minimum level for this logger

def angular_distance_degrees(angle1, angle2):
    """
    Calculates the shortest angular distance in degrees between two angles.

    Args:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.

    Returns:
        float: The shortest angular distance in degrees (unsigned).
    """
    # Calculate the absolute difference between the angles
    difference = abs(angle1 - angle2)

    # Use modulo 360 to handle angles greater than 360 or negative
    # This gives the difference in the range [0, 360)
    wrapped_difference = difference % 360

    # The shortest distance is the smaller of the wrapped difference
    # and 360 minus the wrapped difference
    distance = min(wrapped_difference, 360 - wrapped_difference)

    return distance

def find_consecutive_proximal_points(scan: List[Tuple[int,float, float]], threshold: float=25.0, min_segment_len=4) -> List[
    List[Tuple[float, float]]]:
    """
    Finds groups of consecutive points that are within a distance threshold
    of the immediately preceding point.

    scan: [(quality, angle, radius).... ]
    where 'angle' is in degrees

    Args:
        points: A list of (x, y) tuples.
        threshold: The maximum distance for points to be considered proximal.

    Returns:
        A list of lists, where each inner list is a sequence of consecutive
        proximal points.
    """
    scan = [(x[0], 360.0 - x[1], x[2]) for x in scan if x[2] < SCAN_RADIUS_MM]
    scan = sorted(scan, key=lambda x: x[1])
    points = lidar_readings_to_cartesian(scan)
    if not points or len(points) < 2:
        return []

    # Convert the list of points to a NumPy array for efficient calculation
    points_arr = np.array(points)
    # scan_arr = np.array(scan)

    # Calculate the Euclidean distance between consecutive points
    # This involves shifting the array by one and calculating the distance
    diffs = points_arr[1:] - points_arr[:-1]
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    # Identify where the distance exceeds the threshold
    # These are the breakpoints between sequences
    break_indices = np.where(distances > threshold)[0] + 1
    # print(f"scan: {scan}")
    # print(f"points: {points}")
    # print(f"break indices: {break_indices}")

    # Use the breakpoints to split the array into contiguous segments
    # The start and end indices of these segments represent the proximal sequences
    segments = np.split(points_arr, break_indices)
    # scans = np.split(scan_arr, break_indices)
    # indicies = np.split(np.array(range(0, len(points))), break_indices)

    # Filter out segments that have only one point, as they can't be "consecutive" proximal
    consecutive_segments = [segment.tolist() for segment in segments if len(segment) >= min_segment_len]
    # consecutive_scans = [scan.tolist() for scan in scans if len(scan) >= min_segment_len]
    # consecutive_indices = [index.tolist() for index in indicies if len(index) >= min_segment_len]

    # If the first point is close to the last point then join them assuming that they are the same froup...
    distance_first_last = math.dist(points[0], points[-1])
    if distance_first_last <= threshold:
        consecutive_segments = [consecutive_segments[-1]+consecutive_segments[0]] + consecutive_segments[1:-1]
        # consecutive_scans = [consecutive_scans[-1]+consecutive_scans[0]] + consecutive_scans[1:-1]
        # consecutive_indices = [consecutive_indices[-1]+consecutive_indices[0]] + consecutive_indices[1:-1]

    # scan_data = []
    # for s in consecutive_scans:
    #     # angle = angular_distance_degrees(s[0][1], s[-1][1])
    #     scan_data.append([s[0][1], s[0][2], s[-1][1], s[-1][2]])

    return consecutive_segments


def find_dissimilar_scans(scan_a: List, scan_b: List, threshold: float=2.5):
    """Compare scan_a with scan_b and return a scan from scan_b which is different or an empty list"""
    # print(f"len(scan_a) = {len(scan_a)}; len(scan_b) = {len(scan_b)}")
    if len(scan_a) != len(scan_b):
        return []
    found = []
    for pts_a, pts_b in zip(scan_a, scan_b):
        # diff_distance = [abs(a[2] - b[2]) for a, b in zip(pts_a, pts_b)]
        diff_angle = [abs(a[1] - b[1]) for a, b in zip(pts_a, pts_b)]
        # avg_diff_distance = sum(diff_distance) / len(diff_distance)
        avg_diff_angle = sum(diff_angle) / len(diff_angle)
        # avg_diff_distance = sum(diff_distance) / len(diff_distance)
        # THERE MUST BE A BETTER WAY OF DOING THIS!
        if avg_diff_angle > threshold:
            # print(f"avg_dif_distance = {avg_diff_distance} avg_diff_angle = {avg_diff_angle} "
            #       f"max(diff_distance) = {max(diff_distance)} max_diff_angle = {max(diff_angle)}")
            found.append(pts_b)
    if len(found) != 1:
        return  [len(found)]

    #logger.debug(f"find_dissimilar_scans: {found}")
    return found[0]

if __name__ == "__main__":
# Example Usage:
    points_list = [
        (1, 1), (1.5, 1.5), (2, 2),  # proximal group 1
        (6,6),
        (10, 10), (10.1, 10.1),  # proximal group 2
        (1, 1),  # single point, not a group
        (1.1, 1.1), (1.2, 1.2), (5, 5)  # proximal group 3 (first two), then a break
    ]
    distance_threshold = 1.0

    proximal_groups, jfl = find_consecutive_proximal_points(points_list, distance_threshold)

    print(f"Points List: {points_list}")
    print(f"Distance Threshold: {distance_threshold}")
    print("Consecutive proximal groups found:")
    print(f"Join first and last: {jfl}")
    for group in proximal_groups:
        print(group)
