import numpy as np
from scipy.integrate import quad
import math


def calculate_pendulum_length_from_arc(arc_length, angle_degrees):
    """
    Calculates the length of a pendulum based on the arc it makes.

    Parameters:
    arc_length (float): The length of the swing (in cm or m).
    angle_degrees (float): The angle from the center in degrees (one side).

    Returns:
    float: The length of the pendulum (in same units as arc_length).
    """
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Pendulum length L = s / theta
    if angle_radians == 0:
        return 0

    length = arc_length / angle_radians
    return length

def calculate_pendulum_length(period):
    """
    Calculates the length of a pendulum given its period.
    Formula: L = (g * T^2) / (4 * pi^2)
    """
    g = 9.81  # Acceleration due to gravity in m/s^2
    length = (g * (period ** 2)) / (4 * (math.pi ** 2))
    return length

def calculate_swing_time(L, theta1, theta2, g=9.81):
    """
    Computes time for a pendulum to move from theta1 to theta2 (in radians).
    Assumes motion starts from rest at the maximum angle theta1.
    """

    # Angular acceleration: d^2theta/dt^2 = -(g/L) * sin(theta)
    # Energy conservation gives velocity: dtheta/dt = sqrt(2g/L * (cos(theta) - cos(theta1)))

    def integrand(theta):
        # Time = Integral(1/velocity dtheta)
        return 1.0 / np.sqrt(2.0 * g / L * (np.cos(theta) - np.cos(theta1)))

    # Integrate from theta1 to theta2
    time, _ = quad(integrand, theta1, theta2)
    return abs(time)

if __name__ == '__main__':
    pendulum_length = calculate_pendulum_length_from_arc(s, angle)

    print(f"Arc Length: {s}")
    print(f"Angle: {angle} degrees")
    print(f"Calculated Pendulum Length: {pendulum_length:.4f}")

    # Example usage:

    length = calculate_pendulum_length(2.0)
    print(f"The length of the pendulum is: {length:.4f} meters")

    # --- Example Usage ---
    initial_angle_deg = 45  # degrees
    final_angle_deg = 44
    g = 9.81

    # Convert to radians
    theta1 = np.radians(initial_angle_deg)
    theta2 = np.radians(final_angle_deg)

    swing_time = calculate_swing_time(length, theta1, theta2, g)
    print(f"Time to swing from {initial_angle_deg} to {final_angle_deg} degrees: {swing_time:.4f} seconds")
