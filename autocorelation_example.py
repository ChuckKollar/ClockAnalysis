import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sounddevice as sd
import time

# To analyze a "clock ticking" sound with autocorrelation in Python, you'll record audio (e.g., using
# sounddevice), process it with NumPy for Fast Fourier Transform (FFT) or direct autocorrelation, and identify
# the dominant frequency (the tick's rate) via peak detection in the frequency spectrum or autocorrelation plot,
# revealing the periodic rhythm, though often you'd use FFT for clearer frequency results in real audio

# --- Configuration ---
SAMPLE_RATE = 44100  # Common audio sample rate (Hz)
DURATION = 1  # Seconds to record
CHUNK_SIZE = 1024 # For processing

# --- 1. Record Audio (Simulated for testing, replace with real recording) ---
# In a real scenario, you'd record from a microphone:
# audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
# sd.wait()
# For demo, let's create a synthetic ticking sound (e.g., 1 Hz tick)
t = np.linspace(0., DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
# A noisy sine wave to simulate a tick
synthetic_tick = 0.7 * np.sin(2 * np.pi * 1.0 * t) * np.exp(-5 * t) # 1 Hz
# Add some noise
synthetic_tick += 0.1 * np.random.randn(len(t))
audio_data = synthetic_tick

# --- 2. Calculate Autocorrelation ---
# Using scipy.signal.correlate for direct autocorrelation
# 'full' mode gives results for all possible lags
correlation = signal.correlate(audio_data, audio_data, mode='full')

# Shift the correlation result so lag 0 is in the middle
lags = signal.correlation_lags(len(audio_data), len(audio_data), mode='full')
# Remove the DC component (correlation at lag 0) for better visualization of peaks
# Normalize by the peak at lag 0 for easier peak detection
correlation_normalized = correlation / correlation[len(correlation)//2]

# --- 3. Find the Ticking Period (Lag) ---
# Find the first significant peak after lag 0
# We look for peaks in the positive lags (right side of the 'full' result)
positive_lags_indices = np.where(lags > 0)[0]
peaks, _ = signal.find_peaks(correlation_normalized[positive_lags_indices], height=0.5, distance=int(SAMPLE_RATE/5)) # Adjust height/distance

if len(peaks) > 0:
    # The first significant peak gives the time period (in samples)
    peak_sample_lag = lags[positive_lags_indices[peaks[0]]]
    period_seconds = peak_sample_lag / SAMPLE_RATE
    frequency_hz = 1 / period_seconds
    print(f"Detected Period: {period_seconds:.3f} s (approx {frequency_hz:.2f} Hz)")
else:
    print("No significant tick detected.")

# --- 4. Plotting (Optional) ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, audio_data)
plt.title("Audio Signal (Synthetic Tick)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(lags, correlation_normalized)
plt.axvline(lags[positive_lags_indices[peaks[0]]], color='r', linestyle='--', label=f'{period_seconds:.3f}s') # Mark the first peak
plt.title("Autocorrelation (Normalized)")
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation")
plt.xlim(0, SAMPLE_RATE/2) # Show a reasonable range of lags
plt.legend()
plt.tight_layout()
plt.show()