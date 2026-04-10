import math
import wave
import noisereduce as nr
import numpy as np
from scipy.signal import butter, filtfilt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def _butter_highpass(cutoff, fs, order=5):
    """Designs a Butterworth high-pass filter."""
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_highpass_filter(data, cutoff, fs, order=5):
    """Applies the high-pass filter to audio data (np array)."""
    b, a = _butter_highpass(cutoff, fs, order=order)
    # Apply the filter forward and backward for zero-phase distortion
    y = filtfilt(b, a, data)
    return y

# Define notes in an octave
# These 12 notes are C, C#/D♭, D, D#/E♭, E, F, F#/G♭, G, G#/A♭, A, A#/B♭, and B.
OCTAVE_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq):
    """
    Map the given frequency (Hz) to the closest standard musical note and octave.
    https://www.johndcook.com/blog/2016/02/10/musical-pitch-notation/

    An octave is the musical interval between two notes where the higher note has double the frequency of the
    lower note. It represents the span of eight diatonic notes (e.g., C to the next C), where both notes share
    the same letter name but are in different registers.
    """
    if freq <= 0:
        # Invalid frequency or silence
        return ""

    # Calculate half steps from A4 (440 Hz; standard tuning)
    # A4 is 49th key on a standard 88-key piano
    h = round(12 * math.log2(freq / 440.0))

    # Calculate note index (0-11) and octave
    note_index = (h + 9) % 12  # +9 to align with C as index 0
    octave = (h + 49) // 12 + 1  # Approximate octave calculation
    # Calculate note name and octave
    # 69 is A4 (440Hz) in MIDI notation
    # note_index = (h + 69) % 12
    # octave = (h + 69) // 12 - 1

    return OCTAVE_NOTES[note_index] , octave

def freq_to_note_str(freq):
    note, octave = freq_to_note(freq)
    return f"{note}{octave}"

def write_wav_file(frames, wav_output_file, channels, sample_width, frame_rate):
    """
    Write the audio data in 'frames' to the bas 'wav-output_file' by adding a time stamp.
    Normalize the 'frames' data to make it as loud as possible before clipping, and reduce
    the noise that is present before writing.
    """
    now = datetime.now()
    file_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    wav_output_file_ts = f"{wav_output_file.rstrip('.wav')}_{file_timestamp}.wav"
    with wave.open(wav_output_file_ts, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(frame_rate)
        # Concatinate the frames together as byte objects
        frames_b = b''.join(frames)
        audio_data = np.frombuffer(frames_b, dtype=np.int16)
        max_peak = np.max(np.abs(audio_data))
        # Define the maximum possible value for a 16-bit signed integer
        # This is the ceiling before clipping occurs
        max_int16 = 32767
        # Calculate the scaling factor to bring the max peak to just below max_int16
        # A small safety margin (e.g., 0.99) can be added to prevent potential floating point errors causing clipping
        scaling_factor = (max_int16 * 0.99) / max_peak
        # Apply the scaling factor to all samples
        normalized_audio_data = (audio_data * scaling_factor).astype(np.int16)
        # The 'prop_decrease' parameter adjusts the intensity of the noise reduction (default is 0.8)
        # 'sr' is the sample rate
        reduced_noise_normalized_audio_data = nr.reduce_noise(y=normalized_audio_data, sr=frame_rate,
                                                              prop_decrease=0.8)
        wf.writeframes(reduced_noise_normalized_audio_data.tobytes())
        # wf.writeframes(normalized_audio_data.tobytes())
        # wf.writeframes(frames_b)  # Write all frames at once
    logging.info(f"File '{wav_output_file}' saved successfully.")

def find_target_freq(data_raw, target_freq, rate):
    data = np.frombuffer(data_raw, dtype=np.int16)
    # Calculate FFT and identify dominant frequency
    fft_data = np.abs(np.fft.rfft(data))
    # https://numpy.org/doc/2.1/reference/generated/numpy.fft.rfftfreq.html
    peak_freq = np.fft.rfftfreq(len(data), 1.0 / rate)[np.argmax(fft_data)]

    # Compare peak against target range [7, 8]
    tfw = 50
    target_freq_max = target_freq + tfw
    target_freq_min = target_freq - tfw
    if target_freq_min <= peak_freq <= target_freq_max:
        logging.info(f"MATCH: {peak_freq:.2f} Hz")
        return True
    return False