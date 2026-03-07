#!/usr/bin/env python3

from rplidar import RPLidar, RPLidarException
from lidar.const import startup_lidar
from lidar.find_proximal_points import find_consecutive_proximal_points, find_dissimilar_scans
import numpy as np
import logging
import time

# Configure the root logger
logging.basicConfig(
    level=logging.WARNING, # Set the minimum log level to capture
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

from datetime import datetime

def nanos_str(nanos):
    nanos_remainder = nanos % 1000000000
    seconds = nanos / 1e9
    dt_object = datetime.fromtimestamp(seconds)
    return f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')}.{str(int(nanos_remainder)).zfill(9)}"

# https://thingspeak.mathworks.com/channels/3258476/private_show
# Channel States:  https://thingspeak.mathworks.com/channels/3258476
# RESR API:  https://www.mathworks.com/help/thingspeak/rest-api.html
def thingspeak_url_1(pendulum_period, projected_daily_deviation, pendulum_swing,
                     lidar_readings_hz, pendulum_found_failure_percentage, lidar_restarts,
                     r_squared, pendulum_width):
    """https://thingspeak.mathworks.com/channels/3258476/api_keys"""
    # Pendulum Period (sec/cycle), Projected Daily Deviation (sec/day), Pendulum Swing (mm),
    # Pendulum Swing Computed (mm), Pendulum Found Errors, LIDAR Restarts
    url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
           f"&field1={pendulum_period:.4f}&field2={projected_daily_deviation:.3f}&field3={pendulum_swing:.1f}"
           f"&field4={lidar_readings_hz:.1f}&field5={pendulum_found_failure_percentage:.1f}"
           f"&field6={pendulum_width:.1f}&field8={r_squared:.4f}"
           )
    logging.info(f"pendulum_period: {pendulum_period:.4f} (sec/cycle)"
                 f"; projected_daily_deviation: {projected_daily_deviation:.3f} (sec/day)"
                 f"; pendulum_swing: {pendulum_swing:.1f} (mm)"
                 f"; lidar readings: {lidar_readings_hz:.1f} (Hz)"
                 f"; pendulum_found_failure_percentage: {pendulum_found_failure_percentage:.1f}"
                 f"; lidar_restarts: {lidar_restarts}"
                 f"; R Squared {r_squared:.4f}"
                 f"; Pendulum Width {pendulum_width:.1f}"
                 )
    return url

import requests

def thingspeak_post(url):
    """
    Sending an HTTP POST request to ThingSpeak to update channel data, the server returns a response text that
    indicates the status of the request, specifically how many entries were successfully written.
    Response Text:
    - A Number (e.g., "1", "2", "345"). Meaning: If the post is successful, ThingSpeak returns the Entry ID of the
      newly created data point.
    - "0" Meaning: The update failed. This usually indicates an invalid API key, incorrect URL structure, or that
      the rate limit (maximum one update per second) was exceeded. Free accounts restricted to one update per channel
      every 15 seconds (4 updates/minute).
      In this case we will sleep and retry in 15 seconds. This will take a process out of commission for that time,
      but by adding an extral process this should not be an issue.

    NOTE: With ThingSpeak you can send a maximum of 3 million messages per year (roughly 8,200 messages/day)
    total across all channels. There is a maximum of 4 channels. Each channel is limited to 8 fields of data.
    There is a maximum of 1 message every 15 seconds.
    """
    try:
        response = requests.post(url)
        if response.status_code == 200:
            try:
                response_text_int = int(response.text.strip())
            except ValueError:
                response_text_int = -1
            logging.info(f"ThingSpeak: Data sent OK: {response_text_int}")
            # There should be a more graceful way of handling this...
            if response_text_int == 0:
                # The update failed. Retry one time after the minimum ThingSpeak delay...
                sleep(16)
                logging.info(f"ThingSpeak: Repeating POST for: {url}")
                requests.post(url)
        else:
            logging.error(f"ThingSpeak: Failed to send data. Status code: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"ThingSpeak: Connection failed: {e}")

# This function is called in the main process's result-handling thread
def error_handler(error):
    """
    Callback function to handle exceptions raised by the task_function.

    The callback runs in a separate thread within the main process, not the worker process that failed.
    """
    logging.error(f"[ERROR CALLBACK] An exception occurred in a subprocess: {error}")
    # Create the exc_info tuple manually
    exc_info_tuple = (type(error), error, error.__traceback__)
    logging.error("[ERROR CALLBACK] traceback: ", exc_info=exc_info_tuple)

consecutive_scans_last = None
pendulum_found_failures: int = 0
def find_pendulum_process(scan_w_time):
    """
    This function is used to find the Pendulum (only moving thing) in the LIDAR scan.
    NOTE: any exceptions that happened here will not be propagated to the caller.
    """
    global consecutive_scans_last, pendulum_found_failures
    nanos = scan_w_time[0]
    consecutive_scans = find_consecutive_proximal_points(scan_w_time[1], lidar_scan_radius_mm)
    if consecutive_scans_last is None:
        consecutive_scans_last = consecutive_scans
        return 0, nanos, []
    scan_data_diff = find_dissimilar_scans(consecutive_scans_last, consecutive_scans)
    consecutive_scans_last = consecutive_scans
    scan_data_diff_len = len(scan_data_diff)
    if scan_data_diff_len == 1:
        pendulum_found_failures += 1
    else:
        logging.debug(f"Found pendulum points: {scan_data_diff_len}")
    return 0, nanos, scan_data_diff

from lidar.fit_sine_with_fft_guess import pendulum_equation, sine_function
from lidar.analyze_clock_rate import analyze_clock_rate
from lidar.remove_outliers import remove_outliers_zscore

def pendulum_info_min_process(nano_first_n_last_points_orig, lidar_restarts, processing_time):
    """
    This is used to find information about the pendulum using the time associated with the
    scan and the first (left most) point, and the lsat (right most) point of the pendulum <t, f, l>.
    """
    global pendulum_found_failures
    nano_first_points, outliers = remove_outliers_zscore(nano_first_n_last_points_orig, 1)
    nano_last_points, _ = remove_outliers_zscore(nano_first_n_last_points_orig, 2)
    pendulum_period, t_uniform, theta_uniform, fitted_params, r_squared = pendulum_equation(nano_first_points, 1)
    _, _, theta_uniform_last, _, r_squared_last = pendulum_equation(nano_last_points, 2)
    projected_daily_deviation, _ = analyze_clock_rate(pendulum_period)
    # the swing is how far left and right the pendulum moves based on the LIDAR data
    pendulum_swing = abs(min(theta_uniform) - max(theta_uniform))
    pendulum_width = abs(min(theta_uniform) - min(theta_uniform_last))
    pendulum_width_last = abs(max(theta_uniform) - max(theta_uniform_last))
    logging.warning(f"pendulum_swing: {pendulum_swing}"
                    f"; pendulum_width: {pendulum_width}"
                    f"; pendulum_width_max: {pendulum_width_last}"
                    f"; r_squared: {r_squared}; r_squared_last: {r_squared_last}")
    # theta_uniform_computed = sine_function(t_uniform, *fitted_params)
    # the computed swing is how far left and right the pendulum would move according to the sine_function
    # pendulum_swing_computed = abs(min(theta_uniform_computed) - max(theta_uniform_computed))
    # There are outliers here so we need to understand what they are and later where they are coming from...
    if outliers or abs(projected_daily_deviation) > 600.0:
        logging.warning(f"projected_daily_deviation: {projected_daily_deviation:.3f} (sec/day)"
                        f"; r_squared: {r_squared:.4f}; outliers: {outliers};"
                        f" nano_first_points: {nano_first_points}")
    lidar_readings = pendulum_found_failures + len(nano_first_n_last_points_orig)
    lidar_readings_hz = lidar_readings / processing_time
    pendulum_found_failure_percentage = (pendulum_found_failures / lidar_readings) * 100.0
    pendulum_found_failures = 0
    # This doesn't say how to fix the data, it just says that it is bad and not to use it.
    # See the discussion of R^2 in fit_sine_with_fft_guess:pendulum_equation()
    if r_squared < r_squared_threshold:
        logging.warning(f"Data discarded because R^2 {r_squared:.4f} < threshold of {r_squared_threshold}"
                        f"; pendulum_period: {pendulum_period:.4f} (sec/cycle)"
                        f"; lidar readings: {lidar_readings_hz:.1f} (Hz)")
        thingspeak_post(f"https://api.thingspeak.com/update?api_key={write_api_key}"
                        f"&field4={lidar_readings_hz:.1f}"
                        f"&field5={pendulum_found_failure_percentage:.1f}"
                        f"&field8={r_squared:.4f}"
                        )
        # the empty array signifies that no data was found.
        return 1, []
    thingspeak_post(thingspeak_url_1(pendulum_period, projected_daily_deviation, pendulum_swing,
                                     lidar_readings_hz, pendulum_found_failure_percentage, lidar_restarts,
                                     r_squared, pendulum_width))
    return 1, nano_first_points

def pendulum_info_hr_process(nano_first_points_orig):
    """
    This is used to find information about the pendulum using the time associated with the
    scan and the first (left most) point of the pendulum.
    """
    nano_first_points, outliers = remove_outliers_zscore(nano_first_points_orig, 1)
    pendulum_period, t_uniform, theta_uniform, fitted_params, r_squared = pendulum_equation(nano_first_points, 1)
    projected_daily_deviation, _ = analyze_clock_rate(pendulum_period)
    url = (f"https://api.thingspeak.com/update?api_key={write_api_key}"
           f"&field7={projected_daily_deviation:.3f}"
           )
    logging.info(f"projected_daily_deviation (hr): {projected_daily_deviation:.3f} (sec/day)")
    if r_squared < r_squared_threshold:
        logging.warning(f"Data discarded because R^2: {r_squared} < threshold of {r_squared_threshold}; pendulum_period: {pendulum_period:.4f}; ")
        return 1, []
    thingspeak_post(url)
    return 1, nano_first_points

from typing import List
from multiprocessing import Pool, TimeoutError, get_context, cpu_count
from multiprocessing.pool import AsyncResult
import copy

APPLY_ASYNC_WITH_N = 14.2 * 60.0
def run_scanner(lidar_restarts):
    """
    This function simply grabs data from the LIDAR as fast as it can and sends data for analysis to one of the
    processes in the pool. The first step is to use the 'find_pendulum_process' to identify the part of the scan
    that represents the pendulum. It collects the left most point of each pendulum in an array and when enough
    points are gathered it sends the data to a process that does data cleaning, curve fitting, R^2 analysis, and
    posting to ThingSpeak.

    It is vitally important to spend as little time as possible in this loop or the LIDAR will generate errors. Even
    functions like 'len()' have been removed in favor of keeping a running count of the items in a list. ALL work of
    any nature should be done in subprocesses! In this manner, the LIDAR scanner should run for a day or more without
    seeing a RPLidarException. This was developed on a 2 GHz Quad-Core Intern Core i5 (Macbook Pro) with 16GB of memory.

    As the hours go by the frequency will increase to about 13.7 Hz, and 14.7 Hz.
    """
    global nanos_first_n_last_points_min, nanos_first_n_last_points_min_len, nanos_first_n_last_points_hr, nanos_first_n_last_points_hr_len
    results: List[AsyncResult] = []
    completed_results: List[AsyncResult] = []
    # https://docs.python.org/3/library/multiprocessing.html
    # Use spawn to prevent issues with forking threads
    ctx = get_context('spawn')
    num_proc = cpu_count()
    logging.warning(f"num_proc: {num_proc}")
    # https://pythonspeed.com/articles/python-multiprocessing/
    # maxtasksperchild specifies the number of tasks a worker process can complete before it is terminated and
    # replaced with a new, "fresh" worker process.
    with ctx.Pool(processes=num_proc, maxtasksperchild=100) as pool:
        start_time = time.perf_counter()
        lidar = startup_lidar(lidar_port, lidar_baud_rate, lidar_motor_rpm, logging)
        while True:
            try:
                for scan in lidar.iter_scans(): # (quality, angle, distance)
                    # NOTE: While time.perf_counter() should be accurate to the submicrosecond (10^-6) level, when
                    # scan_with_time is printed we use millisecond accurate (10^-3).
                    scan_with_time = (time.perf_counter(), scan)
                    # Submit a single task to the pool, non-blocking
                    result_obj: AsyncResult = pool.apply_async(find_pendulum_process,
                                                               args=(scan_with_time,),
                                                               error_callback=error_handler
                                                               )
                    results.append(result_obj)

                    for result in results:
                        # Check if the task is complete without blocking the main process...
                        if result.ready() and result not in completed_results:
                            # Non-blocking get: use a very short timeout or check ready() first
                            # The get() will return immediately once ready() is True
                            # This will also prevent a crashed worker from hanging the .get
                            value = result.get(timeout=0.1)
                            if value[0] == 0:
                                # Process the results of 'find_pendulum_process'...
                                value_nanos = value[1]
                                value_scan = value[2]
                                if len(value_scan) > 1:
                                    # Scan info: <time, left_most_point, right_most_point>
                                    nano_first_n_last_point = (value_nanos, value_scan[0][1], value_scan[-1][1])
                                    nanos_first_n_last_points_min.append(nano_first_n_last_point)
                                    nanos_first_n_last_points_min_len += 1
                                    nanos_first_n_last_points_hr.append(nano_first_n_last_point)
                                    nanos_first_n_last_points_hr_len += 1
                                    if nanos_first_n_last_points_hr_len >= APPLY_ASYNC_WITH_N*60.0:
                                        result_obj: AsyncResult = pool.apply_async(pendulum_info_hr_process,
                                                                                   args=(copy.deepcopy(nanos_first_n_last_points_hr),),
                                                                                   error_callback=error_handler
                                                                                   )
                                        results.append(result_obj)
                                        nanos_first_n_last_points_hr = []
                                        nanos_first_n_last_points_hr_len = 0
                                    elif nanos_first_n_last_points_min_len >= APPLY_ASYNC_WITH_N*5.0:
                                        processing_time = time.perf_counter() - start_time
                                        result_obj: AsyncResult = pool.apply_async(pendulum_info_min_process,
                                                                                   args=(copy.deepcopy(nanos_first_n_last_points_min),
                                                                                         lidar_restarts,
                                                                                         processing_time,),
                                                                                   error_callback=error_handler
                                                                                   )
                                        results.append(result_obj)
                                        start_time = time.perf_counter()
                                        nanos_first_n_last_points_min = []
                                        nanos_first_n_last_points_min_len = 0
                                completed_results.append(result)
                            if value[0] == 1:
                                # Process the results of 'pendulum_info_min_process' or 'pendulum_info_hr_process'...
                                completed_results.append(result)
                    completed_results_set = set(completed_results)
                    results = [item for item in results if item not in completed_results_set]
                    completed_results = []
            #  If the worker raises a standard Python exception (rather than a hard crash), that exception is
            #  caught by the pool and re-raised in the main process when AsyncResult.get() is called.
            except RPLidarException as e:
                health = lidar.get_health()
                logging.error(f"RPLidar Exception: {e}; Lidar Health: {health}")
            except TimeoutError as e:
                logging.error(f"Subprocess Exception: {e} out (likely crashed).")
            except KeyboardInterrupt:
                logging.error("Keyboard Interrupt")
            except Exception as e:
                # This handles Python-level exceptions raised by the worker
                logging.error(f"Exception: {e}")
                logging.error("Exception traceback: ", exc_info=(type(e), e, e.__traceback__))
            finally:
                logging.fatal('Stoping...')
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                pool.close()
                pool.join()
                return lidar_restarts+1

import configparser
import os
from time import sleep

nanos_first_n_last_points_min = []
nanos_first_n_last_points_hr = []
nanos_first_n_last_points_min_len = 0
nanos_first_n_last_points_hr_len = 0

# Copy config.ini.example to config.ini and change the WRITE_API_KEY to the one that you get from
# https://thingspeak.mathworks.com/channels/??????/api_keys
config = configparser.ConfigParser()
ini_path = os.path.join(os.getcwd(), 'config.ini')
config.read(ini_path)
write_api_key = config.get('ThingSpeak', 'WRITE_API_KEY').strip('\'"')
lidar_port = config.get('RPLIDAR', 'PORT').strip('\'"')
try:
    lidar_baud_rate: int = int(config.get('RPLIDAR', 'BAUD_RATE').strip('\'"'))
    lidar_motor_rpm: int = int(config.get('RPLIDAR', 'MOTOR_PWM').strip('\'"'))
    lidar_scan_radius_mm: float = float(config.get('RPLIDAR', 'SCAN_RADIUS_MM').strip('\'"'))
    # R_SQUARED_THRESHOLD = 0.7 # .25 was not sensitive enough see fit_sine_with_fft_guess; Typically 0.99?? is seen in logs.
    r_squared_threshold: float = float(config.get('pendulum_info_min_process', 'R_SQUARED_THRESHOLD').strip('\'"'))
except ValueError:
    print("Error reading config.ini; string to number conversion error")
    logging.fatal("Error reading config.ini; string to number conversion error")
    exit(1)

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
# 0) After a few days the program seems to hang without generating any errors, the LIDAR continues to spin.
# Add code to look for possibilities like the processes go away and never finish their computation. This could
# because they get stuck in processing or they generate an exception and we don't hear about it.
# Error during curve fitting: Optimal parameters not found: Number of calls to function has reached maxfev = 1000.
# So, make sure that all of the functions called from this file use logger and not print for errors.
# 1) Need to test if this detects a stopped or bumped pendulum or does the R^2 negate it?
# Stopping the pendulum for a moment results in a slightly larger R^2 (R Squared 0.9300)
# but there is it is possible to detect an anomaly in the 'Pendulum Period', 'Time Keeping',
# and 'Pendulum Swing' (all in the minute average graphs) for that time.
# 2) Need to determine if there is a way to automatically set best thresholds like that of
# find_consecutive_proximal_points, remove_outliers_zscore, and APPLY_ASYNC_WITH_N.
# 3) Why are there angles > 360.0 and 3 min between updates???
# 2026-02-23 09:31:43,757 - INFO - root - ThingSpeak: Data sent OK: 2513
# 2026-02-23 09:34:49,756 - INFO - root - outliers: [(302514.246668439, 404.74745646394194, 1.4349167116675894), (302427.111368239, 468.41218321847646, -9.070645649589546), ...
# 2026-02-23 09:34:49,756 - INFO - root - Data discarded because R^2: 0.18733608583468397 < threshold of 0.7; pendulum_period: 1.9997644948513849;
# 4) Need to get readings closer to clock time; hours close to the hour, and minutes close to the minute.
# 5) Since almost nothing now is written to the log file it would be nice to write an entry every X hours (24?)
