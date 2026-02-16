#!/usr/bin/env python3
from typing import List

import matplotlib
from rplidar import RPLidar, RPLidarException
# This must before importing pyplot
matplotlib.use('Qt5Agg')
from matplotlib import colormaps
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from lidar.const import startup_lidar, SCAN_RADIUS_MM
from lidar.find_proximal_points import find_consecutive_proximal_points
import numpy as np
import time
from enum import Enum
import math
import requests
from lidar.analyze_clock_rate import analyze_clock_rate

# Install this to get the right LIDAR package...
# $ pip install rplidar-roboticia
# Install this for making the movie...
# $ brew update; brew install ffmpeg
# Install this for making the animated graph...
# $ brew install qt@5
# $ pip3 install pyqt5
# Use an interactive backend (e.g., 'Qt5Agg')
# $ pip install requests


class Direction(Enum):
    MOVING_RIGHT = 1
    MOVING_LEFT = 2
    NONE = 3

class Pendulum:
    """Pendulum information"""

    def __init__(self):
        self.pendulums_processed = 0
        self.pendulum_error = 0
        self.pendulum_direction_change_error = 0
        self.max_r_pendulum = None
        self.max_l_pendulum = None
        self.max_angle_unchanged_cnt = 0
        self.direction_current = Direction.NONE
        self.direction_time_changed_to_right = None
        self.pendulum_period = []
        self.pendulum_width = []

        self.last_pendulum = None

    def reset(self):
        self.pendulums_processed = 0
        self.pendulum_error = 0
        self.pendulum_direction_change_error = 0
        self.max_r_pendulum = None
        self.max_l_pendulum = None
        self.max_angle_unchanged_cnt = 0
        self.direction_current = Direction.NONE
        self.direction_time_changed_to_right = None
        self.pendulum_period = []
        self.pendulum_width = []

    def update_max_swing_angles(self, pendulum):
        r_pendulum = pendulum[0]
        l_pendulum = pendulum[-1]

        # Initialize with data from first swing while praying that it's not broken....
        if self.max_r_pendulum is None and self.max_l_pendulum is None:
            self.max_r_pendulum = r_pendulum
            self.max_l_pendulum = l_pendulum
            self.last_pendulum = pendulum
            return

        # Pendulum should move (exceed the current maximums) only in one direction...
        if l_pendulum[1] > self.max_l_pendulum[1] and r_pendulum[1] < self.max_r_pendulum[1]:
            self.pendulum_error += 1
            return
        self.pendulums_processed += 1

        width = calculate_distance(r_pendulum, l_pendulum)
        self.pendulum_width.append(width)

        # Is there a new maximum angle?
        # NOTE: This does not work when crossing 0
        if l_pendulum[1] > self.max_l_pendulum[1]:
            self.max_angle_unchanged_cnt = 0
            self.max_l_pendulum = l_pendulum
            if self.direction_current == Direction.NONE:
                # To bootstrap the change in direction algorithm...
                self.direction_current = Direction.MOVING_LEFT
        elif r_pendulum[1] < self.max_r_pendulum[1]:
            self.max_angle_unchanged_cnt = 0
            self.max_r_pendulum = r_pendulum
            if self.direction_current == Direction.NONE:
                self.direction_current = Direction.MOVING_RIGHT
        else:
            self.max_angle_unchanged_cnt += 1

        # Is there a change in direction?
        direction_changed: Direction = Direction.NONE
        if self.max_l_pendulum[1] > l_pendulum[1] and self.direction_current == Direction.MOVING_LEFT:
            direction_changed = Direction.MOVING_RIGHT
        elif self.max_r_pendulum[1] < r_pendulum[1] and self.direction_current == Direction.MOVING_RIGHT:
            direction_changed = Direction.MOVING_LEFT

        if direction_changed == self.direction_current and self.direction_current is not Direction.NONE:
            self.pendulum_direction_change_error += 1
            print("Direction Error")
        # The period of a pendulum is the time it takes to complete one full swing or oscillation (back and forth)
        # and return to its starting point, measured in seconds.
        elif direction_changed != self.direction_current and direction_changed is not Direction.NONE:
            if direction_changed == Direction.MOVING_RIGHT:
                if self.direction_time_changed_to_right is None:
                    self.direction_time_changed_to_right = time.perf_counter()
                else:
                    # Mark a period when the pendulum moves right...
                    #print(f"delta arc R: {pendulum[0][1]-self.max_r_pendulum[1]:.2f} deg;")
                    now = time.perf_counter()
                    self.pendulum_period.append(now - self.direction_time_changed_to_right)
                    self.direction_time_changed_to_right = now
            #else:
                #print(f"delta arc L: {pendulum[-1][1] - self.max_l_pendulum[1]:.2f} deg;")
            self.direction_current = direction_changed

        # is it time to call it done for this cycle?
        if self.max_angle_unchanged_cnt > 200:
            period = filtered_mean(self.pendulum_period)
            error, behavior = analyze_clock_rate(period)
            print(f"@ pendulums_processed {self.pendulums_processed}; errors {self.pendulum_error}; direction errors: {self.pendulum_direction_change_error}; "
                  f" max_l_angle {self.max_l_pendulum[1]:.2f}; max_r_angle {self.max_r_pendulum[1]:.2f};"
                  f" swing angle {max_angle(self.max_l_pendulum, self.max_r_pendulum):.2f} deg;"
                  f" swing distance: {calculate_distance(self.max_l_pendulum, self.max_r_pendulum):.1f} mm;"
                  f" width: {filtered_mean(self.pendulum_width):.1f} mm;"
                  f" period: {period:.3f} sec;"
                  f" running {behavior} {error:.2f} sec/day;"
                  )
            self.thingsspeak_post()
            # @ pendulums_processed 528; errors 1; max_l_angle 24.95; max_r_angle 337.52; swing angle 47.44 deg; swing distance: 269.3 mm pendulum width: 218.8 mm
            # @ pendulums_processed 369; errors 3; max_l_angle 25.08; max_r_angle 337.66; swing angle 47.42 deg; swing distance: 270.3 mm pendulum width: 216.2 mm
            # @ pendulums_processed 358; errors 3; max_l_angle 25.00; max_r_angle 337.61; swing angle 47.39 deg; swing distance: 268.3 mm pendulum width: 217.3 mm
            # @ pendulums_processed 657; errors 7; max_l_angle 24.94; max_r_angle 337.64; swing angle 47.30 deg; swing distance: 269.0 mm pendulum width: 218.5 mm
            # @ pendulums_processed 907; errors 8; max_l_angle 25.00; max_r_angle 337.47; swing angle 47.53 deg; swing distance: 269.7 mm pendulum width: 215.7 mm
            # @ pendulums_processed 245; errors 9; max_l_angle 24.97; max_r_angle 337.84; swing angle 47.12 deg; swing distance: 267.3 mm pendulum width: 213.7 mm
            self.reset()

        self.last_pendulum = pendulum

    def get_max_angles(self):
        return self.max_r_pendulum[1], self.max_l_pendulum[1], max_angle(self.max_l_pendulum, self.max_r_pendulum)

    # https://thingspeak.mathworks.com/channels/3258476/private_show
    # Channel States:  https://thingspeak.mathworks.com/channels/3258476
    # RESR API:  https://www.mathworks.com/help/thingspeak/rest-api.html
    def things_speak_url(self):
        """https://thingspeak.mathworks.com/channels/3258476/api_keys"""
        write_api_key = ""
        # GET https://api.thingspeak.com/update?api_key=87YUBRFXK5VZOLJG&field1=0
        swing_angle = max_angle(self.max_l_pendulum, self.max_r_pendulum)
        sewing_distance = calculate_distance(self.max_l_pendulum, self.max_r_pendulum)
        period = filtered_mean(self.pendulum_period)
        width = filtered_mean(self.pendulum_width)
        url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
               f"&field1={self.max_l_pendulum[1]}&field2={self.max_r_pendulum[1]}&field3={swing_angle}"
               f"&field4={sewing_distance}&field5={period}&field6={width}")
        return url

    def thingsspeak_post(self):
        try:
            response = requests.post(self.things_speak_url())
            if response.status_code == 200:
                print(f"Data sent successfully. Response: {response.text}")
            else:
                print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}")

def max_angle(a, b):
    """Calculate the maximum angle between two points in degrees"""
    return a[1] + 360 - b[1]

def calculate_distance(a, b):
    """Calculate distance between two points with radius in degrees and distance in millimeters"""
    theta1 = math.radians(a[1])
    theta2 = math.radians(b[1])

    # Apply the Law of Cosines formula
    distance = math.sqrt(a[2] ** 2 + b[2] ** 2 - 2 * a[2] * b[2] * math.cos(theta2 - theta1))
    return distance

def remove_outliers_std(data, num_std=2):
    """Removes data points outside a certain number of standard deviations."""
    data_mean = np.mean(data)
    data_std = np.std(data)

    # Calculate upper and lower bounds
    lower_bound = data_mean - (num_std * data_std)
    upper_bound = data_mean + (num_std * data_std)

    # print(f"Mean: {data_mean:.2f}, Std: {data_std:.2f}")
    # print(f"Keeping values between {lower_bound:.2f} and {upper_bound:.2f}")

    # Filter data
    filtered_data = [x for x in data if x >= lower_bound and x <= upper_bound]
    print(f"mean: {data_mean}, std: {data_std}")
    return filtered_data

def filtered_mean(data):
    return np.mean(remove_outliers_std(data))

IMIN = 0
IMAX = 50
FRAMES = 400

# Error counts...
pendulum_found_failures = 0
lidar_restarts = 0

# Counts...
consecutive_scans_last = None
update_run_time_total = 0
update_run_cnt = 0

# These are global so that the lidar and plot can be restarted if there is an exception...
lidar = None
iterator = None
line = None
ani = None


def find_dissimilar_scans(scan_a: List, scan_b: List, threshold: float=0.75):
    """Compare scan_a with scan_b and return a scan from scan_b which is different or an empty list"""
    global pendulum_found_failures
    # print(f"len(scan_a) = {len(scan_a)}; len(scan_b) = {len(scan_b)}")
    if len(scan_a) != len(scan_b):
        return []
    found = []
    for pts_a, pts_b in zip(scan_a, scan_b):
        # diff_distance = [abs(a[2] - b[2]) for a, b in zip(pts_a, pts_b)]
        diff_angle = [abs(a[1] - b[1]) for a, b in zip(pts_a, pts_b)]
        # avg_diff_distance = sum(diff_distance) / len(diff_distance)
        avg_diff_angle = sum(diff_angle) / len(diff_angle)
        if avg_diff_angle > threshold:
            # print(f"avg_dif_distance = {avg_diff_distance} avg_diff_angle = {avg_diff_angle} "
            #       f"max(diff_distance) = {max(diff_distance)} max_diff_angle = {max(diff_angle)}")
            found.append(pts_b)
    if len(found) != 1:
        pendulum_found_failures += 1
        return []
    return found[0]

# Need to see when doing this faster does... It seems to make it flakey...

def print_global_data():
    print(f"---Runs {update_run_cnt} avg time: {update_run_time_total/update_run_cnt:.3f}"
          f" pendulum_found_failures: {pendulum_found_failures}"
          f" lidar_restarts: {lidar_restarts}")

def update_continuous_plot(_frame):
    """Update function for each scan"""
    global consecutive_scans_last, lidar, iterator, line, ani
    global update_run_cnt, update_run_time_total, lidar_restarts

    start_time = time.perf_counter()
    try:
        scan = next(iterator)  # (quality, angle, distance)
    except RPLidarException:
        print_global_data()
        print("RPLidar Exception caught, attempting to restart and reset...")
        lidar_restarts += 1
        # Orderly shut down...
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()
        # Reconnect logic here...
        # Stop and restart the animation
        ani.event_source.stop()
        # Reset data here if necessary
        ani.event_source.start()
        lidar = startup_lidar()
        iterator = lidar.iter_scans()
        pendulum.reset()
        return line,
    scan = [(x[0], 360.0 - x[1], x[2]) for x in scan if x[2] < SCAN_RADIUS_MM]
    scan = sorted(scan, key=lambda x: x[1])
    _, consecutive_scans, _= find_consecutive_proximal_points(scan)
    if consecutive_scans_last is None:
        consecutive_scans_last = consecutive_scans
        return line,
    scan_data_diff = find_dissimilar_scans(consecutive_scans_last, consecutive_scans)
    consecutive_scans_last = consecutive_scans
    if len(scan_data_diff) == 0:
        update_run_time_total += time.perf_counter() - start_time
        update_run_cnt += 1
        return line,
    pendulum.update_max_swing_angles(scan_data_diff)
    offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan_data_diff])
    line.set_offsets(offsets)
    intens = np.array([meas[0] for meas in scan_data_diff])
    line.set_array(intens)

    update_run_time_total += time.perf_counter() - start_time
    update_run_cnt += 1
    if update_run_cnt % 100 == 0:
        print_global_data()
    return line,

def make_continuous_plot():
    """Animates distances and measurement quality producing a mp4"""
    global consecutive_scans_last, lidar, iterator, line, ani, update_run_time_total
    lidar = startup_lidar()

    try:
        print("Plotting...")
        fig = plt.figure()
        ax = plt.subplot(111, projection='polar')
        cmap = colormaps['Greys_r']
        line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX], cmap=cmap, lw=0)
        ax.set_rmax(SCAN_RADIUS_MM)
        ax.grid(True)

        iterator = lidar.iter_scans()
        consecutive_scans_last = None
        ani = animation.FuncAnimation(fig, update_continuous_plot,
                                      frames=100, interval=20,
                                      cache_frame_data=False,repeat=True)
        plt.show()
    finally:
        print("Done...")
        print("Stopping motor and disconnecting...")
        lidar.stop()
        # The motor must be explicitly stopped
        lidar.stop_motor()
        # The serial connection
        lidar.disconnect()

pendulum = Pendulum()

if __name__ == '__main__':
    make_continuous_plot()
