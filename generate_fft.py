import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def plot_fft_spectrum(filename):
    """
    Reads a WAV file, computes its FFT, and plots the magnitude spectrum.
    """
    # 1. Read the audio file
    # fs_rate is the sampling frequency (samples per second)
    # signal is the raw audio data (array of amplitudes)
    fs_rate, signal = wavfile.read(filename)

    # If the audio is stereo (2 channels), convert it to mono by averaging the channels
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    # 2. Perform the FFT
    # The result is an array of complex numbers
    N = signal.shape[0]  # Total number of samples
    # This performs the core Fast Fourier Transform operation on the signal data.
    fourier_transform = np.fft.fft(signal)

    # 3. Calculate the magnitude of the complex coefficients
    # The magnitude represents the amplitude of each frequency component
    # The output of the FFT is an array of complex numbers. The magnitude (abs()) of these
    # numbers gives the energy or amplitude at each frequency.
    magnitude_spectrum = np.abs(fourier_transform)

    # 4. Generate the corresponding frequency values
    # np.fft.fftfreq returns the frequency bin centers in Hz
    # This function helps generate the actual frequency values in Hertz that correspond to the
    # FFT output bins, using the original sampling rate.
    frequencies = np.fft.fftfreq(N, d=1 / fs_rate)

    # 5. Focus on the positive frequency range (single-sided spectrum)
    # For a real-valued signal, the spectrum is symmetric, so we only need the first half
    half_N = N // 2  # Use integer division for slicing
    frequencies_positive = frequencies[:half_N]
    magnitude_positive = magnitude_spectrum[:half_N]

    # 6. Plot the results
    plt.figure(figsize=(10, 5))

    # Because audio signals are real-valued, their frequency spectrum is symmetric. You typically
    # only plot the first half of the FFT output to show the positive frequencies.

    # Plotting the time-domain signal
    plt.subplot(2, 1, 1)
    time_vector = np.arange(N) / fs_rate
    plt.plot(time_vector, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time Domain Signal")

    # Plotting the frequency spectrum
    plt.subplot(2, 1, 2)
    plt.plot(frequencies_positive, magnitude_positive, 'r')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain Spectrum (FFT)")
    plt.tight_layout()
    plt.show()

# Example usage:
# Make sure you have a 'your_audio_file.wav' file in the same directory
plot_fft_spectrum('audio/Herschede_audio_hr.wav')