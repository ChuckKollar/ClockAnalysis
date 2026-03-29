import pyaudio
import wave
import numpy as np
from scipy.fftpack import fft
import noisereduce as nr
import logging
from datetime import datetime

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG, # Set the minimum log level to capture
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', # Customize the log message format
    filename='./logs/westminster.log', # Log to a file (optional, defaults to console)
    filemode='a' # Append to the log file (default is 'a', 'w' overwrites)
)

def nanos_str(nanos):
    nanos_remainder = nanos % 1000000000
    seconds = nanos / 1e9
    dt_object = datetime.fromtimestamp(seconds)
    return f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')}.{str(int(nanos_remainder)).zfill(9)}"

MAX_ALLOWED_AMP = 0.8  # %
def auto_scale(data_chunk):
    """
    If the microphone is too close to the bells the audio will clip. This prevents that.
    """
    # Convert byte data to numpy float array
    audio_data = np.frombuffer(data_chunk, dtype=np.float32)

    # Check max amplitude
    max_amp = np.max(np.abs(audio_data))

    # Scale if necessary
    if max_amp > MAX_ALLOWED_AMP:
        gain = MAX_ALLOWED_AMP / max_amp
        audio_data = audio_data * gain

    return audio_data.tobytes()


# brew install portaudio
# pip install --upgrade pip
# xcode-select --install
# pip install PyAudio numpy scipy noisereduce

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5  # Analyze 5-second segments

# Westminster notes frequencies (approximate)
# G#4 (415.3 Hz), F#4 (369.99 Hz), E4 (329.63 Hz), B3 (246.94 Hz)
#TARGET_FREQS = [415, 370, 330, 247]

# Westminster Quarters https://en.wikipedia.org/wiki/Westminster_Quarters
# G♯4, F♯4, E4, B3
Q1 = [415.3, 369.99, 329.63, 246.94]
# E4, G♯4, F♯4, B3
Q2 = [329.63, 415.3, 369.99, 246.94]
# E4, F♯4, G♯4, E4
Q3 = [329.63, 369.99, 415.3, 329.63]
# G♯4, E4, F♯4, B3
Q4 = [415.3, 329.63, 369.99, 246.94]
# B3, F♯4, G♯4, E4
Q5 = [246.94, 369.99, 415.3, 329.63]
CHANGE_Q1 = Q1
CHANGE_Q2 = Q2 + Q3
CHANGE_Q3 = Q4 + Q5 + Q1
CHANGE_Q4 = Q2 + Q3 + Q4 + Q5 # + hour (E3)

# The first tones on all Quarters...
FIRST_TONES = [415.3, 329.63, 246.94]

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
        print(f"MATCH: {peak_freq:.2f} Hz")
        return True
    return False

def is_chime(data, chunk, rate):
    now = datetime.now()
    formatted_time_ms = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    logging.debug(f"len data: {len(data)}; time: {formatted_time_ms}")
    # # 1. Denoise the incoming audio
    # # Using noisy audio to estimate noise profile (stationary)
    # reduced_noise = nr.reduce_noise(y=data[:, 0], sr=RATE, stationary=True)
    #
    # # 2. Normalize
    # if np.max(np.abs(reduced_noise)) == 0: return False
    # normalized_audio = reduced_noise / np.max(np.abs(reduced_noise))
    scaled_data = auto_scale(data)

    # Perform FFT
    data_int = np.frombuffer(scaled_data, dtype=np.int16)
    fft_data = np.abs(fft(data_int))[0:chunk // 2]
    freqs = np.fft.fftfreq(chunk, 1.0 / rate)[0:chunk // 2]

    # Simple peak detection
    peak_freq = freqs[np.argmax(fft_data)]

    # Check if the peak is near our target frequencies
    for target in FIRST_TONES:
        if abs(peak_freq - target) < 5:  # 5Hz tolerance
            now = datetime.now()
            formatted_time_ms = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            logging.debug(f"Westminster Chime Detected; target: {target}; time: {formatted_time_ms}")
            print(f"Westminster Chime Detected; target: {target}; time: {formatted_time_ms}")
            return True
    return False

from typing import List
from multiprocessing import Pool, TimeoutError, get_context, cpu_count
from multiprocessing.pool import AsyncResult

# This function is called in the main process's result-handling thread
def error_handler(error):
    """
    Callback function to handle exceptions raised by the task_function.

    The callback runs in a separate thread within the main process, not the worker process that failed.
    """
    logging.error(f"[ERROR CALLBACK] An exception occurred in a subprocess: {error}")
    # Create the exc_info tuple manually
    exc_info_tuple = (type(error), error, error.__traceback__)
    logging.error("[ERROR CALLBACK] traceback: ", exc_info=exc_info_tuple)

def listen_westminster(p):
    # The number of frames (samples) read in each iteration of the loop.
    chunk: int = 2048
    # The sampling rate (samples per second, e.g., 44100 Hz).
    rate = 48000
    stream = None
    stream_o = None
    try:
        stream = p.open(format=FORMAT, channels=1, rate=rate, input=True, frames_per_buffer=chunk,
                        input_device_index=2)
        stream_o = p.open(format=FORMAT, channels=2, rate=rate, output=True, output_device_index=4)
        print("Listening for Westminster Chimes...")
        ctx = get_context('spawn')
        num_proc = cpu_count()
        results: List[AsyncResult] = []
        completed_results: List[AsyncResult] = []
        with ctx.Pool(processes=num_proc, maxtasksperchild=100) as pool:
            while True:
                # seconds_to_read = 1
                # frames = []
                # for i in range(0, int(rate / chunk * seconds_to_read )):
                #     data = stream.read(chunk)
                #     frames.append(data)
                # frames = b''.join(frames)
                frames = stream.read(chunk)
                stream_o.write(frames)
                result_obj: AsyncResult = pool.apply_async(is_chime, args=(frames,chunk,rate,), error_callback=error_handler)
                results.append(result_obj)

                for result in results:
                    # Check if the task is complete without blocking the main process...
                    if result.ready() and result not in completed_results:
                        _ = result.get(timeout=0.1)
                        completed_results.append(result)
                completed_results_set = set(completed_results)
                results = [item for item in results if item not in completed_results_set]
                completed_results = []
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        # This handles Python-level exceptions raised by the worker
        logging.error(f"Exception: {e}")
        logging.error("Exception traceback: ", exc_info=(type(e), e, e.__traceback__))
    finally:
        logging.info("Stopping...")
        stream.stop_stream()
        stream.close()
        stream_o.stop_stream()
        stream_o.close()
        p.terminate()

def listen_for_peaks(p, record_seconds, wav_output_file):
    # The number of frames (samples) read in each iteration of the loop.
    chunk: int = 2048
    # The sampling rate (samples per second, e.g., 44100 Hz).
    sample_rate = 48000
    stream = None
    frames = []
    channels_i = 1
    try:
        stream = p.open(format=FORMAT, channels=channels_i, rate=sample_rate, input=True, frames_per_buffer=chunk,
                        input_device_index=2)
        # Record in chunks for the specified duration
        for i in range(0, int(sample_rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
            data_np = np.frombuffer(data, dtype=np.int16)
            # np.append(frames_np, data_np)
            # Calculate FFT and identify dominant frequency
            fft_data = np.abs(np.fft.rfft(data_np))
            # https://numpy.org/doc/2.1/reference/generated/numpy.fft.rfftfreq.html
            peak_freq = np.fft.rfftfreq(len(data_np), 1.0 / sample_rate)[np.argmax(fft_data)]
            logging.debug(f"Peak: {peak_freq:.2f} Hz")

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        # This handles Python-level exceptions raised by the worker
        logging.error(f"Exception: {e}")
        logging.error("Exception traceback: ", exc_info=(type(e), e, e.__traceback__))
    finally:
        logging.info("Stopping...")
        stream.stop_stream()
        stream.close()
        now = datetime.now()
        file_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        wav_output_file_ts = f"{wav_output_file.rstrip('.wav')}_{file_timestamp}.wav"
        with wave.open(wav_output_file_ts, 'wb') as wf:
            wf.setnchannels(channels_i)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(sample_rate)
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
            reduced_noise_normalized_audio_data = nr.reduce_noise(y=normalized_audio_data, sr=sample_rate, prop_decrease=0.8)
            wf.writeframes(reduced_noise_normalized_audio_data.tobytes())
            #wf.writeframes(normalized_audio_data.tobytes())
            #wf.writeframes(frames_b)  # Write all frames at once
        p.terminate()
        logging.info(f"File '{wav_output_file}' saved successfully.")

if __name__ == '__main__':
    logging.info("Starting...")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i))
    #listen_westminster(p)
    listen_for_peaks(p, 30, './logs/output.wav')
