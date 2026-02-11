# https://github.com/Hyun-je/pyrplidar
from pyrplidar import PyRPlidar
import time

def check_connection():
    info = lidar.get_info()
    print("info :", info)

    health = lidar.get_health()
    print("health :", health)

    samplerate = lidar.get_samplerate()
    print("samplerate :", samplerate)

    scan_modes = lidar.get_scan_modes()
    print("scan modes :")
    for scan_mode in scan_modes:
        print(scan_mode)


def simple_express_scan():
    lidar.set_motor_pwm(500)
    time.sleep(2)

    scan_generator = lidar.start_scan_express(4)

    for count, scan in enumerate(scan_generator()):
        print(count, scan)
        if count == 360: break

    lidar.stop()
    lidar.set_motor_pwm(0)


def simple_scan():
    lidar.set_motor_pwm(500)
    time.sleep(2)

    scan_generator = lidar.force_scan()

    for count, scan in enumerate(scan_generator()):
        print(count, scan)
        if count == 360: break

    lidar.stop()
    lidar.set_motor_pwm(0)


if __name__ == "__main__":
    lidar = PyRPlidar()
    lidar.connect(port="/dev/cu.SLAB_USBtoUART", baudrate=256000, timeout=3)
    # Linux   : "/dev/ttyUSB0"
    # macOS   : "/dev/cu.SLAB_USBtoUART"
    # Windows : "COM5"

    check_connection()
    simple_scan()

    lidar.disconnect()