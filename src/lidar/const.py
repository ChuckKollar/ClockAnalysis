import math
from rplidar import RPLidar
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the minimum level for this logger

# Define the serial port name.
# Be sure to adjust this to your specific system.

_RPLIDAR_PORT_MACOS = '/dev/cu.SLAB_USBtoUART'
_RPLIDAR_PORT_LINUX = '/dev/ttyUSB0'
_RPLIDAR_PORT_WINDOWS = 'COM5'
RPLIDAR_PORT = _RPLIDAR_PORT_MACOS

BAUDRATE = 256000 # 115200 bps default

SCAN_RADIUS_MM = 500.0

DEFAULT_MOTOR_PWM = 660

def _polar_to_cartesian(r, theta):
    """
    Converts single polar coordinates (radius r, angle theta in radians)
    to Cartesian coordinates (x, y).
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

def lidar_readings_to_cartesian(readings):
    """
    Convert the lidar readings to Cartesian coordinates.
    :param readings: [(quality, angle, distance), ...]
    :return:
    """
    return [_polar_to_cartesian(x[2], math.radians(x[1])) for x in readings]

def startup_lidar(logger):
    lidar = RPLidar(RPLIDAR_PORT, BAUDRATE, logger=logger)
    # DEFAULT_MOTOR_PWM = 660 (default)
    #lidar.motor_speed = 330
    lidar.clean_input()

    info = lidar.get_info()
    logger.info(f"Lidar Info: {info}")

    health = lidar.get_health()
    logger.info(f"Lidar Health: {health}")

    lidar.connect()
    lidar.clean_input()

    return lidar
