#!/usr/bin/env python3

from rplidar import RPLidar, RPLidarException
import time
from lidar.const import startup_lidar
from lidar.find_proximal_points import find_consecutive_proximal_points, find_dissimilar_scans

# Install this to get the right LIDAR package...
# $ pip install rplidar-roboticia
# Install this for making the movie...
# $ brew update; brew install ffmpeg
# Install this for making the animated graph...
# $ brew install qt@5
# $ pip3 install pyqt5
# Use an interactive backend (e.g., 'Qt5Agg')
# $ pip install requests

pendulum_found_failures: int = 0
lidar_restarts: int = 0
def print_global_data():
    print(f" pendulum_found_failures: {pendulum_found_failures}"
          f" lidar_restarts: {lidar_restarts}")

from datetime import datetime

def nanos_str(nanos):
    nanos_remainder = nanos % 1000000000
    seconds = nanos / 1e9
    dt_object = datetime.fromtimestamp(seconds)
    return f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')}.{str(int(nanos_remainder)).zfill(9)}"

consecutive_scans_last = None
def find_pendulum_thread(scan_w_time):
    global consecutive_scans_last, pendulum_found_failures
    nanos = scan_w_time[0]
    consecutive_scans = find_consecutive_proximal_points(scan_w_time[1])
    if consecutive_scans_last is None:
        consecutive_scans_last = consecutive_scans
        return 0, nanos, []
    scan_data_diff = find_dissimilar_scans(consecutive_scans_last, consecutive_scans)
    consecutive_scans_last = consecutive_scans
    if len(scan_data_diff) == 1:
        pendulum_found_failures += 1
    return 0, nanos, scan_data_diff

from lidar.fit_sine_with_fft_guess import pendulum_equation, sine_function
from lidar.analyze_clock_rate import analyze_clock_rate
import requests

# https://thingspeak.mathworks.com/channels/3258476/private_show
# Channel States:  https://thingspeak.mathworks.com/channels/3258476
# RESR API:  https://www.mathworks.com/help/thingspeak/rest-api.html
def things_speak_url(write_api_key, pendulum_period, projected_daily_deviation, pendulum_swing, pendulum_swing_computed):
    """https://thingspeak.mathworks.com/channels/3258476/api_keys"""
    # GET https://api.thingspeak.com/update?api_key=87YUBRFXK5VZOLJG&field1=0
    # Pendulum Period (sec/cycle), Projected Daily Deviation (sec/day), Pendulum Swing (mm),
    # Pendulum Swing Computed (mm), Pendulum Found Errors, LIDAR Restarts
    url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
           f"&field1={pendulum_period}&field2={projected_daily_deviation}&field3={pendulum_swing}"
           f"&field4={pendulum_swing_computed}&field5={pendulum_found_failures}"
           f"&field6={lidar_restarts}"
           )
    print(f"pendulum_period: {pendulum_period:.2f}; projected_daily_deviation: {projected_daily_deviation:.2f}"
          f"; pendulum_swing: {pendulum_swing:.2f}; pendulum_swing_computed: {pendulum_swing_computed:.2f}"
          f"; pendulum_found_failures: {pendulum_found_failures:}; lidar_restarts: {lidar_restarts:.2f}")
    return url

def thingsspeak_post(write_api_key, period, error, swing, swing_computed):
    try:
        response = requests.post(things_speak_url(write_api_key, period, error, swing, swing_computed))
        if response.status_code == 200:
            print(f"Data sent OK: {response.text}")
        else:
            print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")

def process_nanos_first_points(nano_first_angles, write_api_key):
    period, t_uniform, theta_uniform, fitted_params = pendulum_equation(nano_first_angles)
    error, _ = analyze_clock_rate(period)
    swing = abs(min(theta_uniform) - max(theta_uniform))
    theta_uniform_computed = sine_function(t_uniform, *fitted_params)
    swing_computed = abs(min(theta_uniform_computed) - max(theta_uniform_computed))
    thingsspeak_post(write_api_key, period, error, swing, swing_computed)
    return 1, nano_first_angles

import multiprocessing
import copy
from time import sleep

def run_scanner(write_api_key):
    global lidar_restarts, consecutive_scans_last
    lidar = startup_lidar()
    consecutive_scans_last = None
    nanos_first_points = []
    try:
        iteration_cnt = 0
        iterator = lidar.iter_scans()
        start_time = time.perf_counter()

        with multiprocessing.Pool(processes=3) as pool:
            results = []
            completed_results = []
            next(iterator) # Throw away the first one
            while True:
                try:
                    scan = next(iterator)  # (quality, angle, distance)
                    scan_with_time = (time.perf_counter(), scan)
                except RPLidarException:
                    lidar_restarts += 1
                    print("RPLidar Exception caught, attempting to restart and reset...")
                    print_global_data()
                    return
                # Submit a single task to the pool, non-blocking
                result_obj = pool.apply_async(find_pendulum_thread, args=(scan_with_time,))
                results.append(result_obj)

                while len(completed_results) < len(results):
                    for result in results:
                        # Check if the task is complete without blocking the main process...
                        if result.ready() and result not in completed_results:
                            # Non-blocking get: use a very short timeout or check ready() first
                            # The get() will return immediately once ready() is True
                            value = result.get(timeout=0.1)
                            if value[0] == 0:
                                value_nanos = value[1]
                                value_scan = value[2]
                                if len(value_scan) > 1:
                                    nano_first_point = (value_nanos, value_scan[0][1], value_scan[0][0])
                                    nanos_first_points.append(nano_first_point)
                                    if len(nanos_first_points) >= 130:
                                        result_obj = pool.apply_async(process_nanos_first_points,
                                                                      args=(copy.deepcopy(nanos_first_points),
                                                                            write_api_key,))
                                        results.append(result_obj)
                                        nanos_first_points = []
                                # print(f"Pendulum {nanos_str(value_nanos)} results[{len(value_scan)}]: {value_scan}", flush=True)
                                completed_results.append(result)
                            if value[0] == 1:
                                # print(f"nano_first_angles: {value[1]}", flush=True)
                                completed_results.append(result)

                # pendulum.update_max_swing_angles(scan_data_diff)

                iteration_cnt += 1
                if iteration_cnt % 30 == 0:
                    # 5.0-5.5 Hz (or readings/swing) no processing; half speed.
                    # 12.9-13.0 Hz no processing; full speed.
                    # 8-10 Hz first processing; full speed.
                    print(f"{30.0/(time.perf_counter() - start_time):.1f} Hz")
                    start_time = time.perf_counter()
    finally:
        print("Done! Stopping motor, disconnecting & closing multiprocessing pool...")
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        pool.close()
        pool.join()
import configparser
import os

# When a new process starts using the 'spawn' method, it re-imports the main script.
# Any code at the global scope that is not protected by an if __name__ == '__main__': block will be executed during
# this re-import process, which can lead to infinite loops of spawning new processes or other errors.
if __name__ == '__main__':
    config = configparser.ConfigParser()
    ini_path = os.path.join(os.getcwd(), 'config.ini')
    config.read(ini_path)
    write_api_key = config.get('ThingSpeak', 'WRITE_API_KEY')
    while True:
        run_scanner(write_api_key)
        sleep(5)
