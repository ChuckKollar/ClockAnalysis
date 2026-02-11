import numpy as np
from scipy.optimize import leastsq
#from const import lidar_readings_to_cartesian

def _fit_circle(points):
    """Fits a circle to 2D points using least squares. Returns (xc, yc, r)."""
    x = points[:, 0]
    y = points[:, 1]

    # Initial guess for center (mean of points)
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """Calculate distance of each point from center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """Calculate algebraic distance from points to circle center c"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Optimize for center
    center_estimate = x_m, y_m
    center, _ = leastsq(f_2, center_estimate)

    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()

    # Calculate residual to determine quality of fit
    residu = np.sum((Ri - R) ** 2)
    return xc, yc, R, residu


def is_on_arc(points, threshold=0.1):
    """Checks if points lie on an arc within a residual threshold."""
    if len(points) < 3: return False, (0, 0, 0)

    xc, yc, r, residu = _fit_circle(np.array(points))

    # If residual is low, points fit a circle well
    is_circle = (residu / len(points)) < threshold

    # Further check: points should not make a full circle
    # (Optional: verify arc angular span if needed)

    return is_circle, (xc, yc, r)


def get_equation_coefficients(x_points, y_points, degree):
    """
    Fits a polynomial of a specified degree to the given x and y points
    and returns the coefficients.

    Args:
        x_points (list): A list or array of x-coordinates.
        y_points (list): A list or array of y-coordinates.
        degree (int): The degree of the polynomial to fit (e.g., 1 for linear, 2 for quadratic).

    Returns:
        numpy.ndarray: An array of coefficients, ordered from the
                       highest power (x^degree) to the constant term (x^0).
    """
    # Convert lists to NumPy arrays for use with polyfit
    x = np.array(x_points)
    y = np.array(y_points)

    # Use numpy.polyfit to get the coefficients
    coefficients = np.polyfit(x, y, degree)

    return coefficients

if __name__ == "__main__":
    # Use the 'annotate.py' utility to make_hover_over_plot() to get a sample of these points...
    # Fudge the threshold till it works....
    # Points lie on an arc. Center: (673.54, -32.99), Radius: 319.93 threshold: 3.0
    # Quadratic Coefficients (a, b, c): [ 7.36852991e-01 -5.27313697e+02  9.42990698e+04]
    # Quadratic Equation: y = 0.74x^2 + -527.31x + 94299.07
    scan = [(15, 342.75, 382.75), (15, 343.546875, 376.0), (15, 344.421875, 374.5), (15, 345.421875, 370.75), (15, 346.4375, 368.0), (15, 347.96875, 365.25), (15, 348.5625, 362.0), (15, 350.03125, 360.75), (15, 350.53125, 359.0), (15, 351.65625, 356.75), (15, 353.03125, 356.5), (15, 354.078125, 356.5), (15, 354.671875, 355.75), (15, 356.28125, 358.25), (15, 357.28125, 359.0), (15, 358.015625, 355.0), (15, 359.125, 351.25), (15, 0.359375, 353.5), (15, 1.5, 355.25), (15, 2.46875, 356.5), (15, 3.234375, 357.5), (15, 4.734375, 360.5), (15, 6.140625, 364.0), (15, 7.1875, 366.25), (15, 8.0625, 369.75), (15, 9.171875, 373.0)]

    # ADD MORE POINTS REPRESENTING THE PENDULUM....

    data_points = np.array(lidar_readings_to_cartesian(scan))
    is_arc, circle_params = is_on_arc(data_points, 3.0)
    if is_arc:
        print(f"Points lie on an arc. Center: ({circle_params[0]:.2f}, {circle_params[1]:.2f}), Radius: {circle_params[2]:.2f}")
        x = [point[0] for point in data_points]
        y = [point[1] for point in data_points]
        coefficients_quadratic = get_equation_coefficients(x, y, 2)
        print(f"Quadratic Coefficients (a, b, c): {coefficients_quadratic}")
        print(f"Quadratic Equation: y = {coefficients_quadratic[0]:.2f}x^2 + {coefficients_quadratic[1]:.2f}x + {coefficients_quadratic[2]:.2f}")
    else:
        print("Points do not form an arc.")
