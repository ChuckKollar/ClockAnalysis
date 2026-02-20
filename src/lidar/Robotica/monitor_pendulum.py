#!/usr/bin/env python3

from rplidar import RPLidar, RPLidarException
from lidar.const import startup_lidar
from lidar.find_proximal_points import find_consecutive_proximal_points, find_dissimilar_scans
import logging
import time

# Configure the root logger
logging.basicConfig(
    level=logging.INFO, # Set the minimum log level to capture
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', # Customize the log message format
    filename='../logs/monitor_pendulum.log', # Log to a file (optional, defaults to console)
    filemode='a' # Append to the log file (default is 'a', 'w' overwrites)
)

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

from datetime import datetime

def nanos_str(nanos):
    nanos_remainder = nanos % 1000000000
    seconds = nanos / 1e9
    dt_object = datetime.fromtimestamp(seconds)
    return f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')}.{str(int(nanos_remainder)).zfill(9)}"

import requests

# https://thingspeak.mathworks.com/channels/3258476/private_show
# Channel States:  https://thingspeak.mathworks.com/channels/3258476
# RESR API:  https://www.mathworks.com/help/thingspeak/rest-api.html
def things_speak_url_1(pendulum_period, projected_daily_deviation, pendulum_swing,
                       pendulum_swing_computed, lidar_restarts, r_squared):
    """https://thingspeak.mathworks.com/channels/3258476/api_keys"""
    # Pendulum Period (sec/cycle), Projected Daily Deviation (sec/day), Pendulum Swing (mm),
    # Pendulum Swing Computed (mm), Pendulum Found Errors, LIDAR Restarts
    url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
           f"&field1={pendulum_period}&field2={projected_daily_deviation}&field3={pendulum_swing}"
           f"&field4={pendulum_swing_computed}&field5={pendulum_found_failures}"
           f"&field6={lidar_restarts}&field8={r_squared}"
           )
    logging.info(f"pendulum_period: {pendulum_period:.2f} (sec/cycle)"
                 f"; projected_daily_deviation: {projected_daily_deviation:.2f} (sec/day)"
                 f"; pendulum_swing: {pendulum_swing:.2f} (mm)"
                 f"; pendulum_swing_computed: {pendulum_swing_computed:.2f} (mm)"
                 f"; pendulum_found_failures: {pendulum_found_failures}"
                 f"; lidar_restarts: {lidar_restarts}"
                 f"; R Squared {r_squared:.4f}")
    return url

def thingsspeak_post(url):
    try:
        response = requests.post(url)
        if response.status_code == 200:
            logging.info(f"Data sent OK: {response.text}")
        else:
            logging.error(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Connection failed: {e}")

consecutive_scans_last = None
def find_pendulum_process(scan_w_time):
    """
    This function is used to find the Pendulum (only moving thing) in the LIDAR scan.
    NOTE: any exceptions that happened here will not be propagated to the caller.
    """
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
from lidar.remove_outliers import remove_outliers_zscore

R_SQUARED_THRESHOLD = 0.4 # .25 was not sensitive enough see fit_sine_with_fft_guess
def pendulum_info_min_process(nano_first_angles_orig, lidar_restarts):
    """
    This is used to find information about the pendulum using the time associated with the
    scan and the first (left most) point of the pendulum.
    """
    # Outliers are harder to spot on the depth value (2) rather than the lateral value (1)
    # because the swing is greater than the front to back thickness of the pendulum.
    nano_first_angles, outliers = remove_outliers_zscore(nano_first_angles_orig, 1)
    pendulum_period, t_uniform, theta_uniform, fitted_params, r_squared = pendulum_equation(nano_first_angles)
    projected_daily_deviation, _ = analyze_clock_rate(pendulum_period)
    # the swing is how far left and right the pendulum moves based on the LIDAR data
    pendulum_swing = abs(min(theta_uniform) - max(theta_uniform))
    theta_uniform_computed = sine_function(t_uniform, *fitted_params)
    # the computed swing is how far left and right the pendulum would move according to the sine_function
    pendulum_swing_computed = abs(min(theta_uniform_computed) - max(theta_uniform_computed))
    # There are outliers here so we need to understand what they are and later where they are coming from...
    if outliers or abs(projected_daily_deviation) > 600.0 :
        logging.info(f"outliers: {outliers}; nono_first_angles: {nano_first_angles}")
    # This doesn't say how to fix the data, just to determine that it was bad and not to use it.
    # See discussion of R^2 in fit_sine_with_fft_guess:pendulum_equation()
    if r_squared < R_SQUARED_THRESHOLD:
        logging.info(f"Data discarded because R^2: {r_squared} < threshold of {R_SQUARED_THRESHOLD};"
                     f" pendulum_period: {pendulum_period}; ")
        thingsspeak_post(f"https://api.thingspeak.com/update?api_key={write_api_key}&field8={r_squared}")
        return 1, []
    thingsspeak_post(things_speak_url_1(pendulum_period, projected_daily_deviation, pendulum_swing,
                                        pendulum_swing_computed, lidar_restarts, r_squared))
    return 1, nano_first_angles

def pendulum_info_hr_process(nano_first_angles_orig):
    """
    This is used to find information about the pendulum using the time associated with the
    scan and the first (left most) point of the pendulum.
    """
    nano_first_angles, outliers = remove_outliers_zscore(nano_first_angles_orig, 1)
    pendulum_period, t_uniform, theta_uniform, fitted_params, r_squared = pendulum_equation(nano_first_angles)
    projected_daily_deviation, _ = analyze_clock_rate(pendulum_period)
    url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
           f"&field7={projected_daily_deviation}"
           )
    logging.info(f"; projected_daily_deviation (hr): {projected_daily_deviation:.2f} (sec/day)")
    if r_squared < R_SQUARED_THRESHOLD:
        logging.info(f"Data discarded because R^2: {r_squared} < threshold of {R_SQUARED_THRESHOLD}; pendulum_period: {pendulum_period}; ")
        return 1, []
    thingsspeak_post(url)
    return 1, nano_first_angles

import multiprocessing
import copy
import traceback

APPLY_ASYNC_WITH_N = 13.4 * 60.0
ITERATION_N = 60
def run_scanner(lidar_restarts):
    global consecutive_scans_last, nanos_first_points_min, nanos_first_points_hr
    iteration_cnt = 0
    results = []
    completed_results = []
    with multiprocessing.Pool(processes=3) as pool:
        while True:
            lidar = startup_lidar(logging)
            consecutive_scans_last = None
            start_time = time.perf_counter()
            try:
                for scan in lidar.iter_scans(): # (quality, angle, distance)
                    scan_with_time = (time.perf_counter(), scan)
                    # Submit a single task to the pool, non-blocking
                    result_obj = pool.apply_async(find_pendulum_process, args=(scan_with_time,))
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
                                        nanos_first_points_min.append(nano_first_point)
                                        nanos_first_points_hr.append(nano_first_point)
                                        # ThingSpeak enforces a minimum interval of 15 seconds between updates to a
                                        # channel's data. So if there is an hour one send it and throw away the minute
                                        # one.
                                        if len(nanos_first_points_hr) >= APPLY_ASYNC_WITH_N*60.0:
                                            result_obj = pool.apply_async(pendulum_info_hr_process,
                                                                          args=(copy.deepcopy(nanos_first_points_hr),))
                                            results.append(result_obj)
                                            nanos_first_points_hr = []
                                        elif len(nanos_first_points_min) >= APPLY_ASYNC_WITH_N:
                                            result_obj = pool.apply_async(pendulum_info_min_process,
                                                                          args=(copy.deepcopy(nanos_first_points_min),
                                                                                lidar_restarts,))
                                            results.append(result_obj)
                                            nanos_first_points_min = []
                                    # print(f"Pendulum {nanos_str(value_nanos)} results[{len(value_scan)}]: {value_scan}", flush=True)
                                    completed_results.append(result)
                                if value[0] == 1:
                                    # print(f"nano_first_angles: {value[1]}", flush=True)
                                    completed_results.append(result)
                    iteration_cnt += 1
                    if iteration_cnt % ITERATION_N == 0:
                        # 5.0-5.5 Hz (or readings/swing) no processing; half speed.
                        # 12.9-13.0 Hz no processing; full speed.
                        logging.info(f"{ITERATION_N / (time.perf_counter() - start_time):.1f} Hz")
                        start_time = time.perf_counter()
            except RPLidarException as e:
                health = lidar.get_health()
                logging.error(f"RPLidar Exception: {e}; Lidar Health: {health}")
                lidar_restarts += 1
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                return lidar_restarts
            except KeyboardInterrupt:
                logging.error('Stoping...')
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                return lidar_restarts

import configparser
import os
from time import sleep

nanos_first_points_min = []
nanos_first_points_hr = []

# Copy config.ini.example to config.ini and change the WRITE_API_KEY to the one that you get from
# https://thingspeak.mathworks.com/channels/??????/api_keys
config = configparser.ConfigParser()
ini_path = os.path.join(os.getcwd(), 'config.ini')
config.read(ini_path)
write_api_key = config.get('ThingSpeak', 'WRITE_API_KEY').strip('\'"')

# When a new process starts using the 'spawn' method, it re-imports the main script.
# Any code at the global scope that is not protected by an if __name__ == '__main__': block will be executed during
# this re-import process, which can lead to infinite loops of spawning new processes or other errors.
if __name__ == '__main__':
    print("Starting...")
    lidar_restarts: int = 0
    while True:
        consecutive_scans_last = None
        sleep(5)
        # this needs to be a local and not a global because it needs to be passed to another process
        lidar_restarts = run_scanner(lidar_restarts)
