#!/usr/bin/env python3

import numpy as np
from lidar.const import startup_lidar, lidar_readings_to_cartesian, SCAN_RADIUS_MM
from lidar.least_squares import is_on_arc
import time

TESTS_CNT = 100

def get_scan_in_readings_about_center(iterator, readings_about_center):
    """
    Retrieve the scan using the iterator and subset it to useful points for this purpose.
    """
    scan = next(iterator)  # (quality, angle, distance)
    # get scans that are within CAN_RADIUS_MM and reverse its direction to meke it consistent with plotting
    scan = [(x[0], 360.0 - x[1], x[2]) for x in scan if x[2] < SCAN_RADIUS_MM]
    scan = sorted(scan, key=lambda x: x[1]) # ascending angle values
    # only get readings that lie within a radius of eadings_about_center
    scan = scan[:readings_about_center] + scan[-readings_about_center:]
    return scan

def find_pendulum_arc(iterator, readings_about_center, pendulum_width, threshold):
    """
    Look for an arc pendulum_width in the scan.
    NOTE: CHANGE TO LOOK FOR PENDULUM_WIDTH WITHIN ARC WITH POSSIBLY MULTIPLE ARCS STARTING WHERE ARCS LEAVE OFF
    RATHER TRY TO CONNECT THE ARCS.
    ALSO, POSSIBLY DO THIS SEVERAL TIMES AND TAKE THE MAX, OR ONLY READINGS NEAR THE CENTER LINE!?
    """
    scan = get_scan_in_readings_about_center(iterator, readings_about_center)
    print(f"Scan Readings: {len(scan)}")
    scan_cartesian = np.array(lidar_readings_to_cartesian(scan))
    for k in range(0, readings_about_center * 2 - pendulum_width):
        datapoints = scan_cartesian[k: k + pendulum_width]
        is_arc, circle_params = is_on_arc(datapoints, threshold)
        if is_arc:
            pendulum_arc_in_scan = scan[k: k + pendulum_width]
            # x = [cord[2] for cord in pendulum_arc_in_scan]
            # pendumum_width_degrees = pendulum_arc_in_scan[-1][1] - pendulum_arc_in_scan[0][1]
            # print(f"find_pendulum_arc Angle: {pendumum_width_degrees} max x diff: {max(x)-min(x)}")
            return pendulum_arc_in_scan
    return None

class Swing:
    def __init__(self, swing, direction, time):
        self.swing = swing
        self.time = time

def monitor_pendulum():
    readings = [(0,0)]
    reading_i = 0
    lidar = None
    try:
        lidar = startup_lidar()
        iterator = lidar.iter_scans()
        # Throw away the first scan because the motor is not to be up to speed...
        next(iterator)
        while True:
            # There is a problem with the lidar_start() built into this....
            readings_about_center, pendulum_width = (25, 14)  # determine_search_parameters()
            last_reading = None
            last_direction = None
            current: Swing = None
            readings_stored = 0
            while True:
                pendulum_arc_in_scan = find_pendulum_arc(iterator, readings_about_center, pendulum_width)
                if pendulum_arc_in_scan is None:
                    break # recalibrate
                elif last_reading is None:
                    last_reading = (pendulum_arc_in_scan[0][1], time.perf_counter())
                elif last_direction is None:
                    last_direction = ((last_reading[0] - pendulum_arc_in_scan[0][1])  + 180) % 360 - 180
                else:
                    direction = ((last_reading[0] - pendulum_arc_in_scan[0][1]) + 180) % 360 - 180
                    # Returns True if both numbers have the same sign or are moving in the same direction
                    if direction * last_direction > 0:
                        # Still moving in the same direction
                        last_reading = (pendulum_arc_in_scan[0], time.perf_counter())
                    else:
                        # The direction of the pendulum has changed
                        readings[reading_i][0 if last_direction > 0 else 1] = last_reading
                        swing_distance_degrees = ((readings[reading_i][0][0] - readings[reading_i][1][0]) + 180) % 360 - 180
                        print(f"Swing distance (deg): {swing_distance_degrees}")
                        readings_stored += 1
                        if readings_stored % 2 == 0:
                            reading_i += 1
                            readings[reading_i] = (0, 0)
                        last_reading = (pendulum_arc_in_scan[0], time.perf_counter())
                        last_direction = direction
    finally:
        print("Stopping motor and disconnecting...")
        lidar.stop()
        # The motor must be explicitly stopped
        lidar.stop_motor()
        # The serial connection
        lidar.disconnect()


def determine_search_parameters(readings_about_center = 40, pendulum_width= 35, threshold = 12):
    """"
    The purpose of this function is to locate the parameters to find the pendulum from the LIDAR readings.
    It begins by assuming that the LIDAR unit is pointed in the direction (roughly) of the pendulum. This
    involves making sure that the power cord is in the opposide side of the pendulum. So, the pendulum will be
    seen by the LIDAR as being around 0 degrees.

    A state machine is used to find the size of the pendulum (in terms of readings), and the size of the swing
    of the pendulum (also in terms of readings). With this information it is possible to search a minimum of data
    points to get the pendulum location. That is this routine should be called with the FINAL readings_about_center and
    pendulum_width. If this fails, then something has gone wrong, and the function should be again run without
    parameters to locate the pendulum again (i.e., the device may have been moved).

    The state machine works as follows:
    (state==1) begins with very small value of the pendulum_width and a large value of
    the readings_about_center and the function increases them until it fails to find a convex arc within the window.
    This is generally because the pendulum_width has been exceeded. The assumption here is that the pendulum will be no
    smaller than the initial pendulum_width and that it will always be found in initial readings_about_center.
    (state==2). Then backs off on pendulum_width till it finds the pendulum again
    (state==3). Then back down on readings_about_center till there is a failure then increases the readings_about_center
    one.

    Each of these tests must succeed TESTS_CNT times.

    The entire search can take on the order of 3 minutes, smaller if the initial guesses are close, but that risks
    not finding the values. You should error on the size of a large initial readings_about_center and a small
    pendulum_width.

    With my Herschede Tall Case clock I get the following results:
    FINAL readings_about_center: 41, pendulum_width: 36
    """
    state = 1

    lidar = None
    start_time = time.perf_counter()
    try:
        lidar = startup_lidar()
        iterator = lidar.iter_scans()
        # Throw away the first scan because the motor is not to be up to speed...
        next(iterator)
        while True:
            found_arc_cnt = 0
            for i in range(TESTS_CNT):
                pendulum_arc_in_scan = find_pendulum_arc(iterator, readings_about_center, pendulum_width, threshold)
                if pendulum_arc_in_scan is not None:
                    found_arc_cnt += 1
            if found_arc_cnt == TESTS_CNT:
                print(f"PASS state={state} readings_about_center: {readings_about_center}, pendulum_width: {pendulum_width}")
                if state == 1:
                    # readings_about_center += 1
                    pendulum_width += 1
                if state == 2:
                    state = 3
                if state == 3:
                    readings_about_center -= 1
            else:
                print(f"FAIL state={state} readings_about_center: {readings_about_center}, pendulum_width: {pendulum_width}")
                if state == 1:
                    state = 2
                if state == 2:
                    pendulum_width -= 1
                if state == 3:
                    readings_about_center += 1
                    break
    finally:
        print(f"FINAL readings_about_center: {readings_about_center}, pendulum_width: {pendulum_width}")
        print("Stopping motor and disconnecting...")
        lidar.stop()
        # The motor must be explicitly stopped
        lidar.stop_motor()
        # The serial connection
        lidar.disconnect()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {int(elapsed_time//60)}:{int(elapsed_time%60)}")
        return readings_about_center, pendulum_width

if __name__ == '__main__':
    determine_search_parameters()