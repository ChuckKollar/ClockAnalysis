#!/usr/bin/env python3
from typing import List

import matplotlib
from rplidar import RPLidar, RPLidarException
# This must before importing pyplot
matplotlib.use('Qt5Agg')
from matplotlib import colormaps
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from lidar.const import startup_lidar, SCAN_RADIUS_MM
from lidar.find_proximal_points import find_consecutive_proximal_points
import numpy as np

# Install this to get the right LIDAR package...
# $ pip install rplidar-roboticia
# Install this for making the movie...
# $ brew update; brew install ffmpeg
# Install this for making the animated graph...
# $ brew install qt@5
# $ pip3 install pyqt5
# Use an interactive backend (e.g., 'Qt5Agg')

IMIN = 0
IMAX = 50
FRAMES = 400
MP4_FILE = 'animate.mp4'
colors = ["red", "green", "brown", "blue", "orange", "purple", "pink", "cyan", "magenta", "yellow"]

def update(_frame, iterator, line):
    """Update function for each frame of the movie"""
    scan_raw = next(iterator) # (quality, angle, distance)
    # throw away measurements that are further away than a meter
    # also the LIDAR and the plotting graph's polar coordinates go in opposite directions
    # so correct for this by reversing the angle
    scan = [(x[0], 360.0 - x[1], x[2]) for x in scan_raw if x[2] < SCAN_RADIUS_MM]
    # Sort so that the angles are increasing....
    scan = sorted(scan, key=lambda x: x[1])
    print("scan[", len(scan), "] [(q, a, d)...] ", scan)
    offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
    line.set_offsets(offsets)
    intens = np.array([meas[0] for meas in scan])
    line.set_array(intens)
    return line,

def make_movie():
    """Animates distances and measurement quality producing a mp4"""
    lidar = startup_lidar()

    try:
        print("Plotting...")
        fig = plt.figure()
        ax = plt.subplot(111, projection='polar')
        cmap = colormaps['Greys_r']
        line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX], cmap=cmap, lw=0)
        ax.set_rmax(SCAN_RADIUS_MM)
        ax.grid(True)

        # lidar.clean_input()
        iterator = lidar.iter_scans()
        # Throw away the first scan because the motor is not to be up to speed...
        next(iterator)
        print('Gathering animation data ', FRAMES, ' frames')
        ani = animation.FuncAnimation(fig, update, fargs=(iterator, line), frames=FRAMES, interval=20, cache_frame_data=False)
        print("Saving animation as animate.mp4 to ", MP4_FILE)
        # Adjust fps to match the interval (1000/interval)
        ani.save(MP4_FILE, writer='ffmpeg', fps=20)
        print("Saving complete.")
    except ValueError as e:
        print(f"Error saving file {MP4_FILE}: {e}")
        print("Please ensure you have FFmpeg installed and accessible in your system's PATH.")
        # Fallback to show the animation in a window if saving fails
        plt.show()
        # plt.close(fig)  # Close the figure as it is already saved
    finally:
        print("Done...")
        print("Stopping motor and disconnecting...")
        lidar.stop()
        # The motor must be explicitly stopped
        lidar.stop_motor()
        # The serial connection
        lidar.disconnect()



def update_annot(ind, annot, scatter):
    """Updates the annotation text and position for the hovered point."""
    pos = scatter.get_offsets()[ind["ind"][0]]
    # Matplotlib's polar axes use theta (angle in radians) for x and r (radius) for y
    theta_rad = pos[0]
    r = pos[1]

    # Convert theta to degrees for easier reading if desired
    theta_deg = np.degrees(theta_rad)
    if theta_deg > 360:
        theta_deg -= 360

    annot.xy = pos
    # Format the text to display polar coordinates (radius, angle)
    text = f"R: {r:.2f}\nAngle: {theta_deg:.2f}°"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)

def hover(event, annot, scatter, fig, ax):
    """Handles the mouse movement event to show/hide the annotation."""
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, annot, scatter)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

def color_arcs(consecutive_indices, arc_colors):
    """Assign a different color to each arc"""
    color_i = 0
    for x in consecutive_indices:
        for i in x:
            arc_colors[i] = colors[color_i]
        color_i += 1
        if color_i >= len(colors):
            color_i = 0

def make_hover_over_plot():
    """
    Make an interactive plot from one scan with mouseover points showing point information.
    Divide the points into arcs (consecutive_indices) and color each arc differently on the plot.
    Create a list (scan_data) containing the first and last point in each arc.
    Return the
    """
    lidar = startup_lidar()

    try:
        print("Plotting...")
        fig = plt.figure()
        ax = plt.subplot(111, projection='polar')
        _ = colormaps['Greys_r']

        # lidar.clean_input()
        iterator = lidar.iter_scans()
        # Throw away the first scan because the motor is not to be up to speed...
        next(iterator)

        scan = next(iterator)  # (quality, angle, distance)
        # throw away measurements that are further away than SCAN_RADIUS_MM
        # also the LIDAR and the plotting graph's polar coordinates go in opposite directions
        # so correct for this by reversing the angle
        scan = [(x[0], 360.0 - x[1], x[2]) for x in scan if x[2] < SCAN_RADIUS_MM]
        # Sort so that the angles are increasing....
        scan = sorted(scan, key=lambda x: x[1])
        # Convert degrees to radians
        x = [np.radians(meas[1]) for meas in scan]
        y = [meas[2] for meas in scan]
        consecutive_indices, _, scan_data = find_consecutive_proximal_points(scan)
        # scan_data: [[343.890625, 327.25, 21.984375, 318.0], [33.078125, 493.5, 86.0625, 232.5], [282.71875, 268.5, 333.53125, 497.75]]
        # scan_data: [[341.015625, 335.25, 16.734375, 314.5], [33.015625, 492.25, 86.828125, 231.25], [283.421875, 265.5, 333.84375, 498.0]]
        # scan_data: [[352.46875, 318.5, 24.53125, 326.75], [33.140625, 493.5, 86.890625, 232.5], [282.515625, 265.5, 333.625, 497.5]]
        # scan_data: [[346.515625, 325.25, 17.53125, 315.5], [33.046875, 497.5, 86.515625, 233.5], [283.21875, 265.5, 333.921875, 498.75]]
        # scan_data: [[341.921875, 335.0, 18.984375, 317.75], [33.0625, 494.25, 87.671875, 230.5], [282.984375, 265.25, 333.578125, 494.5]]
        # scan_data: [[350.125, 323.0, 26.609375, 328.25], [33.015625, 492.5, 86.6875, 232.5], [282.875, 265.5, 333.546875, 494.5]]
        # scan_data: [[347.375, 324.5, 18.921875, 313.75], [33.015625, 494.25, 87.078125, 232.25], [282.21875, 265.25, 333.671875, 496.75]]
        # scan_data: [[340.28125, 339.75, 16.546875, 314.25], [33.328125, 492.75, 86.546875, 232.25], [282.5625, 265.5, 333.71875, 497.5]]

        # Assign a default color to every point in the scan...
        arc_colors = ["black" for _ in range(len(x))]
        # Do curve matching and associate the points of every curve with a different color
        # colors = ["red", "green", "brown", "blue", "orange", "purple", "pink", "cyan", "magenta", "yellow"]
        print(f"scan_data: {scan_data}")
        color_arcs(consecutive_indices, arc_colors)
        # find_and_color_arcs_radial(list(zip(y, x)), arc_colors, colors)
        line = ax.scatter(np.array(x), np.array(y), s=4, c=arc_colors)
        ax.set_rmax(SCAN_RADIUS_MM)
        ax.grid(True)

        # 4. Set up the annotation box (tooltip)
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5),
                            arrowprops=dict(arrowstyle="->"))

        annot.set_visible(False)
        # 5. Connect the hover event to the function
        # The 'motion_notify_event' is triggered when the mouse moves
        fig.canvas.mpl_connect("motion_notify_event", lambda event: hover(event, annot, line, fig, ax))

    finally:
        print("Done...")
        print("Stopping motor and disconnecting...")
        lidar.stop()
        # The motor must be explicitly stopped
        lidar.stop_motor()
        # The serial connection
        lidar.disconnect()
        # 6. Display the plot
        print('Close the window to exit the program')
        # Since this will hang till the user closes the window, make sure that the LIDAR motor is stopped first
        plt.show()


if __name__ == '__main__':
    make_hover_over_plot()
    # make_movie()
