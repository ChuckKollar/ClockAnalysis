#!/usr/bin/env python3
"""Measures sensor scanning speed"""
from rplidar import RPLidar
import time
from lidar.const import RPLIDAR_PORT, BAUDRATE

PORT_NAME = '/dev/ttyUSB0'

def run():
    lidar = RPLidar(RPLIDAR_PORT, BAUDRATE)
    old_t = None
    data = []
    cycles = 0
    try:
        print('Press Ctrl+C to stop')
        for _ in lidar.iter_scans():
            now = time.time()
            if old_t is None:
                old_t = now
                continue
            delta = now - old_t
            print('%.2f Hz, %.2f RPM' % (1/delta, 60/delta))
            data.append(delta)
            cycles += 1
            old_t = now
    except KeyboardInterrupt:
        print('Stoping. Computing mean...')
        lidar.stop()
        lidar.disconnect()
        sum_data = sum(data)
        delta = sum_data/len(data)
        print(f"Mean: {1/delta:.2f} Hz (cycles/sec), {60/delta:.2f} RPM")

if __name__ == '__main__':
    run()