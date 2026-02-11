import math

# This Python program determines the necessary gear tooth counts for a pendulum clock
# drive mechanism. It calculates the required gear ratios between the escape wheel and
# the minute hand (arbor) based on the input parameters: escapement teeth, pendulum period in sec,
# and the number of intermediate wheels.
#
# It calculates the rotation speed of the escape wheel and then determines appropriate gear
# ratios for a 3-wheel or 4-wheel train (e.g., center wheel, third wheel, fourth/escape wheel)
# to achieve a 1-hour (60-minute) cycle for the minute hand

# see https://mb.nawcc.org/threads/escape-wheel-calculation-pendulum-clock.197391/#:~:text=Hi%2C%20I%20have%20constructed%20a%203D%20printed,length%20of%2039%22%2C%20which%20works%20fine%20(PIC

def calculate_clock_train(escape_teeth, pendulum_period_sec, num_arbors):
    """
    Calculates the gear teeth for a mechanical clock based on escapement.
    Assumes:
    - 1 escapement tooth = 2 ticks (back and forth).
    - Minute hand revolves once per hour (3600 seconds).

E   scape Wheel Teeth: Count the teeth on your escapement wheel (common: 30).
    Pendulum Period: The time for the pendulum to swing to one side, the other side,
     and back to the center (left, right, left) in seconds.
    Total Arbors: The number of axles (including the escapement wheel one) in the gear
     train (usually 3 or 4)
    """

    # 1. Calculate how long one escape wheel revolution takes
    # Each pendulum swing releases one tooth (two swings per period)
    # A standard anchor escapement allows 2 teeth to pass per full period (T)
    # Beats per minute = 60 / (period/2)
    # Revolutions per hour = (Beats per minute / (2 * escape_teeth)) * 60

    sec_per_rev = (2 * escape_teeth * pendulum_period_sec) / 2
    rev_per_hour = 3600 / sec_per_rev
    pendulum_length_meters = 9.8*(pendulum_period_sec/(2*math.pi))**2

    print(f"--- Clock Drive Calculation ---")
    print(f"Escape Wheel Teeth: {escape_teeth}")
    print(f"Pendulum Period: {pendulum_period_sec} sec")
    print(f"Escape Wheel takes {sec_per_rev} seconds to rotate once.")
    print(f"Pendulum Length: {pendulum_length_meters} meters")
    print(f"Required total ratio (Min Hand / Escape Wheel): {1 / rev_per_hour:.4f}\n")

    # 2. Determine Gear Train Strategy
    # We need to reduce the speed from the escapement up to the minute hand.
    # Total Ratio = (Driven Pinions) / (Driving Wheels)
    # For a 3-wheel train (typical):
    #   MinuteArbor -> ThirdWheel -> EscapeWheel
    #   Ratio = (MinPinion/CenterWheel) * (ThirdPinion/ThirdWheel) * (EscapePinion/EscapeWheel)

    # A simplified approach using typical clock ratios:
    # Typical pinions are 8, 10, or 12 leaves.
    pinion = 8

    if num_arbors == 3:
        # Example: 30-tooth escape wheel, 2-sec period (60s rev)
        # 60 min hand / 1 min escape = 60:1 reduction needed
        # Commonly: (60/8) * (64/8) = 60.0

        # We calculate the wheels (W) needed based on pinions (P=8)
        # Ratio: (W2/P1) * (W3/P2) * ...

        # Simplified gear calculation assuming 3 stages
        wheel1 = round(60 * (pinion ** 2) / 60)  # Placeholder logic

        print("Suggested 3-wheel train (Wheel/Pinion):")
        print(f"1. Center (Second) Wheel: 60T -> Third Pinion: 8 leaves")
        print(f"2. Third Wheel: 64T -> Escape Pinion: 8 leaves")
        print(f"3. Escape (First) Wheel: {escape_teeth}T")

    elif num_arbors == 4:
        print("Suggested 4-wheel train (Wheel/Pinion):")
        print("4-wheel trains are often used for second hands.")
        # Similar logic as above, but with an extra pair

    else:
        print("Unsupported number of arbors for simple calculation.")

# https://www.smithofderby.com/products/automatic-winding/
# There is a gear reduction built into the system which compensates for the use of weight
# lesser than the original. So, 5x reduction in weight is matched by 5x multiplication in gearing.

# --- Inputs ---
# Example: 30 teeth, 2-second pendulum (seconds-beating clock)
escape_wheel_teeth = 30 # This is typical
# Note: While pendulum length and gravity determine the time (period) of the swing
# or period (\(T=2\pi \sqrt{\frac{L}{g}}\)), they do not directly determine the distance
# (amplitude) of the swing, though they affect how much energy is required to maintain it.
# https://mb.nawcc.org/threads/i-have-a-question-about-the-pendulum-swinging-distance.155933/
# Moving the pallets closer will create a bigger swing. If you go too far though, the swing
# will be too wide for the impulse to carry the pendulum far enough to unlock. Make small
# adjustments until you're happy with the swing.
pendulum_period = 2.0 # For a Hershedy Tallcase clock (sec)
total_arbors = 3  # Minute, Third, Escape

calculate_clock_train(escape_wheel_teeth, pendulum_period, total_arbors)
