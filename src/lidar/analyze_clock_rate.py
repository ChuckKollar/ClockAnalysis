import math


def analyze_clock_rate(measured_period, ideal_period=2.0):
    """
    Analyzes how a clock will perform based on its pendulum period.
    Ideal period is 2.0s for one full cycle (tick-tock) in most grandfather clocks.
    """
    # Calculate error in seconds per day
    # There are 86400 seconds in a day.
    # Total ticks in a day = 86400 / ideal_period

    seconds_per_day = 86400
    total_ticks = seconds_per_day / ideal_period

    # Error per swing in seconds
    error_per_swing = measured_period - ideal_period

    # Total error in seconds per day
    total_error_spd = error_per_swing * total_ticks

    # Result interpretation
    if abs(total_error_spd) < 0.1:
        status = "Accurate"
    elif total_error_spd > 0:
        status = "Fast"
    else:
        status = "Slow"

    return total_error_spd, status

if __name__ == '__main__':
    # --- Example Usage ---
    # Suppose a clock meant to have a 2.0s period is measured at 2.001s
    target = 2.0
    measured = 1.9823137226339895

    error, behavior = analyze_clock_rate(measured)

    print(f"Target Period: {target} s")
    print(f"Measured Period: {measured} s")
    print(f"Clock is {behavior}.")
    print(f"Error: {error:.2f} seconds per day.")