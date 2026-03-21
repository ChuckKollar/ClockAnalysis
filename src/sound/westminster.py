import pyaudio
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
CHUNK = 2048
RATE = 48000
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

def is_chime(data):
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
    fft_data = np.abs(fft(data_int))[0:CHUNK // 2]
    freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE)[0:CHUNK // 2]

    # Simple peak detection
    peak_freq = freqs[np.argmax(fft_data)]

    # Check if the peak is near our target frequencies
    for target in FIRST_TONES:
        if abs(peak_freq - target) < 10:  # 10Hz tolerance
            now = datetime.now()
            formatted_time_ms = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            logging.debug(f"Westminster Chime Detected; time: {formatted_time_ms}")
            print(f"Westminster Chime Detected; time: {formatted_time_ms}")
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
    stream = None
    stream_o = None
    try:
        stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK,
                        input_device_index=2)
        stream_o = p.open(format=FORMAT, channels=2, rate=RATE, output=True, output_device_index=4)
        print("Listening for Westminster Chimes...")
        ctx = get_context('spawn')
        num_proc = cpu_count()
        results: List[AsyncResult] = []
        completed_results: List[AsyncResult] = []
        with ctx.Pool(processes=num_proc, maxtasksperchild=100) as pool:
            while True:
                data = stream.read(CHUNK)
                stream_o.write(data)
                result_obj: AsyncResult = pool.apply_async(is_chime, args=(data,), error_callback=error_handler)
                results.append(result_obj)

                for result in results:
                    # Check if the task is complete without blocking the main process...
                    if result.ready() and result not in completed_results:
                        value = result.get(timeout=0.1)
                        if value is True:
                            print("Westminster Chime Detected!")
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

if __name__ == '__main__':
    logging.info("Starting...")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i))
    listen_westminster(p)
