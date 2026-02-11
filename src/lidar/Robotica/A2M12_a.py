# https://github.com/Roboticia/RPLidar  (see the /examples directory)
from rplidar import RPLidar
from lidar.const import RPLIDAR_PORT, BAUDRATE

# Remember to install the package:
# $ pip install rplidar-roboticia

# https://static.generation-robots.com/media/pj2-ld310-slamtec-rplidar-datasheet-a2m12-v1-0-en-2874.pdf
# http://bucket.download.slamtec.com/ccb3c2fc1e66bb00bd4370e208b670217c8b55fa/LR001_SLAMTEC_rplidar_protocol_v2.1_en.pdf


def run_lidar_scan():
    """
    Connects to the RPLIDAR, retrieves data, and prints the measurements.
    """
    # Initialize the RPLidar object with the correct port and baudrate (115200 for A2M12).
    lidar = RPLidar(RPLIDAR_PORT, BAUDRATE)

    try:
        # Get and print device information
        info = lidar.get_info()
        print(f"Lidar Info: {info}")
        # Lidar Info: {'model': 44, 'firmware': (1, 32), 'hardware': 6, 'serialnumber': '9DF8ECF0C3E09ED2A0EA98F30E204110'}

        # Get and print device health
        health = lidar.get_health()
        print(f"Lidar Health: {health}")
        # Lidar Health: ('Good', 0)

        print("Starting scan...")

        # Iterate over scans and print data.
        # Get complete 360-degree scans one by one. This is the most efficient way to get live data.
        # Each scan is a list of measurements: (quality, angle, distance).
        # angle (degree) Current heading angle of the measurement
        # distance (mm) Current measured distance value between the rotating core of the RPLIDAR and the sampling point
        # quality (u8) Measurement quality (0 ~ 255)
        # start_flag (Bool) Flag of a new scan
        # checksum The Checksum of RPLIDAR return data
        scans = lidar.iter_scans()
        for i, scan in enumerate(scans):
            print(f"Scan {i}: Got {len(scan)} measurements")

            # Example of how to access individual points in a scan:
            #
            for quality, angle, distance in scan:
                print(f"Angle: {angle:.2f} degrees, Distance: {distance:.2f} mm, Quality: {quality}")

            # Stop after 10 scans
            if i >= 10:
                break

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        print("Stopping due to KeyboardInterrupt")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop the motor and disconnect the sensor in the finally block
        # to ensure cleanup even if an error occurs.
        print("Stopping motor and disconnecting...")
        lidar.stop()
        # The motor must be explicitly stopped
        lidar.stop_motor()
        # The serial connection
        lidar.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    run_lidar_scan()