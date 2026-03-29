import numpy as np


def calculate_pythagorean_tuning(reference_freq=440.0):
    """
    Calculates Pythagorean tuning scale (12 notes) based on a reference frequency.
    Pythagorean tuning uses a ratio of 3:2 for perfect fifths.
    """
    # Create the 12-note chromatic scale ratios based on 5ths
    # Order: F, C, G, D, A, E, B, F#, C#, G#, D#, A#
    # Middle 'A' is note 4 (0-indexed) in this sequence
    fifths = np.array([-4, -1, 2, -3, 0, 3, -2, 1, 4, -1, 2, -5])

    # Calculate ratios using 3/2 ^ n, reduced by octaves (2^m)
    ratios = []
    for f in fifths:
        ratio = (3 / 2) ** f
        # Keep the ratio within the 1-2 range (an octave)
        while ratio < 1:
            ratio *= 2
        while ratio >= 2:
            ratio /= 2
        ratios.append(ratio)

    # Sort ratios to be in order from C to B
    ratios = sorted(ratios)

    # Calculate frequencies from ratios, adjusting to reference_freq
    # Assume the reference frequency is A4
    reference_note_index = 9  # Assuming sorted list (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)

    # For A4=440, we need to map the ratios to the correct octave
    # This simplified version gives the frequencies within the 400-800Hz range
    # based on the 1-2 ratio span.

    tuning_scale = {}
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # To match A=440 as the 10th note (index 9)
    # 440 is the reference.
    # We normalize such that the ratio of A is 1, then scale.
    a_ratio = ratios[9]

    for i, ratio in enumerate(ratios):
        freq = (ratio / a_ratio) * reference_freq
        tuning_scale[note_names[i]] = round(freq, 2)

    return tuning_scale


# Example Usage
if __name__ == "__main__":
    # https://pytuning.readthedocs.io/_/downloads/en/0.7.3/pdf/
    # You can change 440.0 to any frequency to change the base pitch, such as 432
    tuning = calculate_pythagorean_tuning(440)
    print("Pythagorean Tuning (A=440Hz):")
    for note, freq in tuning.items():
        print(f"{note}: {freq} Hz")