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

import requests

def thingsspeak_post(url):
    """
    Send data to the chart.
    Sending a HTTP POST request to ThingSpeak to update channel data, the server returns a response text that
    indicates the status of the request, specifically how many entries were successfully written.
    Response Text:
    - A Number (e.g., "1", "2", "345"). Meaning: If the post is successful, ThingSpeak returns the Entry ID of the
      newly created data point.
    - "0" Meaning: The update failed. This usually indicates an invalid API key, incorrect URL structure, or that
      the rate limit (maximum one update per second) was exceeded.
    """
    try:
        response = requests.post(url)
        if response.status_code == 200:
            logging.info(f"ThingSpeak: Data sent OK: {response.text}")
        else:
            logging.error(f"ThingSpeak: Failed to send data. Status code: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"ThingSpeak: Connection failed: {e}")

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

R_SQUARED_THRESHOLD = 0.7 # .25 was not sensitive enough see fit_sine_with_fft_guess; Typically 0.99?? is seen in logs.
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
    if outliers or abs(projected_daily_deviation) > 600.0 or r_squared < R_SQUARED_THRESHOLD:
        logging.info(f"outliers: {outliers}; nono_first_angles: {nano_first_angles}")
    # This doesn't say how to fix the data, just to determine that it was bad and not to use it.
    # See discussion of R^2 in fit_sine_with_fft_guess:pendulum_equation()
    if r_squared < R_SQUARED_THRESHOLD:
        logging.info(f"Data discarded because R^2: {r_squared} < threshold of {R_SQUARED_THRESHOLD};"
                     f" pendulum_period: {pendulum_period}; ")
        thingsspeak_post(f"https://api.thingspeak.com/update?api_key={write_api_key}&field8={r_squared}")
        # the empty array signifies that no data was found.
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
    logging.info(f"projected_daily_deviation (hr): {projected_daily_deviation:.2f} (sec/day)")
    if r_squared < R_SQUARED_THRESHOLD:
        logging.info(f"Data discarded because R^2: {r_squared} < threshold of {R_SQUARED_THRESHOLD}; pendulum_period: {pendulum_period}; ")
        return 1, []
    thingsspeak_post(url)
    return 1, nano_first_angles

from typing import List
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
import copy
import traceback

APPLY_ASYNC_WITH_N = 13.2 * 60.0
ITERATION_N = 60
def run_scanner(lidar_restarts):
    """
    This function simply grabs data from the LIDAR as fast as it can and sends data for analysis to one of the
    processes in the pool. The first step is to use the 'find_pendulum_process' to identify the part of the scan
    that represents the pendulum. It collects the left most point of each pendulum in an array and when enough
    points are gathered it sends the data to a process that does data cleaning, curve fitting, R^2 analysis and
    posting to ThingSpeak.

    It is vitally important to spend as little time as possible in this loop. Even functions like 'len()' have
    been removed in favor of keeping a running count of the items in a list. It should be possible to go for one
    hour of more without seeing a RPLidarException. ALL work of any nature should be done in subprocesses!
    """
    global nanos_first_points_min, nanos_first_points_min_len, nanos_first_points_hr, nanos_first_points_hr_len
    iteration_cnt: int = 0
    results: List[AsyncResult] = []
    completed_results: List[AsyncResult] = []
    # https://docs.python.org/3/library/multiprocessing.html
    with Pool(processes=4) as pool:
        lidar = startup_lidar(logging)
        start_time = time.perf_counter()
        while True:
            try:
                for scan in lidar.iter_scans(): # (quality, angle, distance)
                    scan_with_time = (time.perf_counter(), scan)
                    # Submit a single task to the pool, non-blocking
                    result_obj: AsyncResult = pool.apply_async(find_pendulum_process, args=(scan_with_time,))
                    results.append(result_obj)

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
                                    nanos_first_points_min_len += 1
                                    nanos_first_points_hr.append(nano_first_point)
                                    nanos_first_points_hr_len += 1
                                    # ThingSpeak enforces a minimum interval of 15 seconds between updates to a
                                    # channel's data. So if there is an hour one send it and throw away the minute
                                    # one.
                                    if nanos_first_points_hr_len >= APPLY_ASYNC_WITH_N*60.0:
                                        result_obj: AsyncResult = pool.apply_async(pendulum_info_hr_process,
                                                                                   args=(copy.deepcopy(nanos_first_points_hr),))
                                        results.append(result_obj)
                                        nanos_first_points_hr = []
                                        nanos_first_points_hr_len = 0
                                    elif nanos_first_points_min_len >= APPLY_ASYNC_WITH_N:
                                        result_obj: AsyncResult = pool.apply_async(pendulum_info_min_process,
                                                                                   args=(copy.deepcopy(nanos_first_points_min),
                                                                                         lidar_restarts,))
                                        results.append(result_obj)
                                        nanos_first_points_min = []
                                        nanos_first_points_min_len = 0
                                completed_results.append(result)
                            if value[0] == 1:
                                completed_results.append(result)
                    completed_results_set = set(completed_results)
                    results = [item for item in results if item not in completed_results_set]
                    completed_results = []
                    iteration_cnt += 1
                    if iteration_cnt % ITERATION_N == 0:
                        # 5.0-5.5 Hz (or readings/swing) no processing; half speed.
                        # 12.9-13.0 Hz no processing; full speed.
                        logging.info(f"{ITERATION_N / (time.perf_counter() - start_time):.1f} Hz")
                        start_time = time.perf_counter()
            except RPLidarException as e:
                if e == 'Check bit not equal to 1':
                    #lidar.reset()
                    # Try just tossing this data and continuing on with the next data packet....
                    continue
                health = lidar.get_health()
                logging.error(f"RPLidar Exception: {e}; Lidar Health: {health}")
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                pool.close()
                pool.join()
                return lidar_restarts+1
            except KeyboardInterrupt:
                logging.error('Stoping...')
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                pool.close()
                pool.join()
                return lidar_restarts

import configparser
import os
from time import sleep

nanos_first_points_min = []
nanos_first_points_hr = []
nanos_first_points_min_len = 0
nanos_first_points_hr_len = 0

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
        # this needs to be a local and not a global because it needs to be passed to another process
        lidar_restarts = run_scanner(lidar_restarts)
        sleep(2)

# TODO:
# 1) Need to test if this detects a stopped or bumped pendulum or does the R^2 negate it?
# 2) Need to determine if there is a way to detect what to set thresholds like that of
# find_consecutive_proximal_points and remove_outliers_zscore.
# 3) Need to find out why the LIDAR generates so many errors.
