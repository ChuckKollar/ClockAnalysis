#!/usr/bin/env python3

import pyaudio
import wave
import numpy as np
from scipy.fftpack import fft
from datetime import datetime
from scipy.signal import find_peaks
import noisereduce as nr
import logging
import argparse
import librosa
import librosa.display
import sounddevice as sd
from sound_utils import freq_to_note, write_wav_file, apply_highpass_filter, OCTAVE_NOTES
import matplotlib.pyplot as plt

# xcode-select --install
# brew install portaudio
# pip install --upgrade pip
# python3.11 -m pip install --upgrade pip
# brew install llvm
# export LLVM_CONFIG=$(brew --prefix llvm)/bin/llvm-config
# pip install PyAudio numpy scipy noisereduce librosa

# brew install miniforge
# exec $SHELL -l
# conda install -c conda-forge librosa

# Configure the root logger
logging.basicConfig(
    level=logging.INFO, # Set the minimum log level to capture
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

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5  # Analyze 5-second segments

# Westminster notes frequencies (approximate)
# G#4 (415.3 Hz), F#4 (369.99 Hz), E4 (329.63 Hz), B3 (246.94 Hz)
#TARGET_FREQS = [415, 370, 330, 247]
# E3 in Pythagorean Tuning https://tunableapp.com/notes/e3/pythagorean/
E3 = 164.814
# Westminster Quarters https://en.wikipedia.org/wiki/Westminster_Quarters
# G♯4, F♯4, E4, B3
Q1 = [415.3, 369.99, 329.63, 246.94]
Q1n = ["G♯4", "F♯4", "E4", "B3"]
# E4, G♯4, F♯4, B3
Q2 = [329.63, 415.3, 369.99, 246.94]
Q2n = ["E4", "G♯4", "F♯4", "B3"]
# E4, F♯4, G♯4, E4
Q3 = [329.63, 369.99, 415.3, 329.63]
Q3n = ["E4", "F♯4", "G♯4", "E4"]
# G♯4, E4, F♯4, B3
Q4 = [415.3, 329.63, 369.99, 246.94]
Q4n = ["E4", "F♯4", "G♯4", "E4"]
# B3, F♯4, G♯4, E4
Q5 = [246.94, 369.99, 415.3, 329.63]
Q5n = ["E4", "F♯4", "G♯4", "E4"]
CHANGE_Q1 = Q1n
CHANGE_Q2 = Q2n + Q3n
CHANGE_Q3 = Q4n + Q5n + Q1n
CHANGE_Q4 = Q2n + Q3n + Q4n + Q5n # + hour (E3)

# The first tones on all Quarters...
FIRST_TONES = [415.3, 329.63, 246.94]


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
        write_wav_file(frames, wav_output_file, channels_i, p.get_sample_size(FORMAT), sample_rate)
        p.terminate()

def listen_for_peaks_in_file(p, wav_input_file):
    # The number of frames (samples) read in each iteration of the loop.
    chunk: int = 2048
    cutoff_hz = 100.0
    wf = None
    try:
        wf = wave.open(wav_input_file, 'rb')
        sample_rate_hz = wf.getframerate()
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()  # bytes per sample
        logging.info(f"Processing file: {wav_input_file} sample_rate_hz: {sample_rate_hz}; n_channels: {n_channels}; samp_width: {samp_width}")
        frames_read_total = 0
        while True:
            current_time_seconds = frames_read_total / float(sample_rate_hz)
            data = wf.readframes(chunk)
            if not data:
                break
            data_np = np.frombuffer(data, dtype=np.int16)

            data_np = nr.reduce_noise(y=data_np, sr=sample_rate_hz, prop_decrease=0.75)
            data_np = apply_highpass_filter(data_np, cutoff_hz, sample_rate_hz, order=5)
            frames_read_total += len(data) // (n_channels * samp_width)  # number of frames in the chunk

            # np.append(frames_np, data_np)
            # Calculate FFT and identify dominant frequency
            # fft_data = np.abs(np.fft.rfft(data_np))
            # https://numpy.org/doc/2.1/reference/generated/numpy.fft.rfftfreq.html
            # peak_freq = np.fft.rfftfreq(len(data_np), 1.0 / sample_rate_hz)[np.argmax(fft_data)]
            # logging.debug(f"Time: {current_time_seconds:.4f} sec; Peak: {peak_freq:.2f} Hz; Note: {freq_to_note_str(peak_freq)}")

            samples = len(data_np)  # Number of samples in the segment
            # 3. Perform Fast Fourier Transform (FFT)
            yf = np.fft.rfft(data_np)  # FFT for real-valued signals
            xf = np.fft.rfftfreq(samples, 1 / sample_rate_hz)  # Frequency bins
            # Get magnitude and normalize
            freq_magnitude = np.abs(yf)
            # 4. Find frequency peaks
            # Use scipy.signal.find_peaks to identify prominent frequencies
            # You might need to adjust the height and distance parameters based on your audio
            peak_indices, _ = find_peaks(freq_magnitude, height=0.01 * np.max(freq_magnitude), distance=10)
            detected_frequencies = xf[peak_indices]
            detected_magnitudes = freq_magnitude[peak_indices]
            # Choose a reference amplitude (e.g., the max value in the spectrum or 1.0 for float data)
            reference_amplitude = np.max(detected_magnitudes)  # This makes the peak 0 dB
            # Convert Amplitude to dB
            # https://stackoverflow.com/questions/53761077/from-amplitude-or-fft-to-db
            # Use a small value (like 1e-10) to avoid log10(0) which results in -inf
            epsilon = 1e-10
            detected_magnitudes_db = 20 * np.log10(detected_magnitudes / reference_amplitude + epsilon)
            # Sort by magnitude for easier reading
            sorted_peaks = sorted(zip(detected_frequencies, detected_magnitudes_db), key=lambda x: x[1], reverse=True)
            str = ""
            first_str = True
            for freq, mag in sorted_peaks[:10]:
                # Big Ben (Hour Bell): Low E (approx. 55 Hz). 30dB is 99.9% sound pressure level (SPL) reduction
                if mag > -30.0:
                    # Only keep frequencies above a threshold
                    if not first_str:
                        str +=", "
                    note, octave = freq_to_note(freq)
                    str += f"{freq:.1f}Hz {note}{octave}"
                    if mag < 0.0:
                        # 0 dB assumed
                        str += f" {mag:.1f}dB"
                    first_str = False
            logging.info(f"Time: {current_time_seconds:.4f}s; {str}")

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        # This handles Python-level exceptions raised by the worker
        logging.error(f"Exception: {e}")
        logging.error("Exception traceback: ", exc_info=(type(e), e, e.__traceback__))
    finally:
        logging.info("Stopping...")
        wf.close()
        p.terminate()

def identify_note_pattern(file_path):
    # 1. Load the audio file (mono=True converts stereo to mono)
    y, sr = librosa.load(file_path, sr=None)

    # 2. Extract Chroma Features (Energy for each of the 12 musical notes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 3. Find the dominant note index for each time frame
    # Index 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    note_indices = np.argmax(chroma, axis=0)

    # 4. Map indices to note names
    raw_notes = [OCTAVE_NOTES[i] for i in note_indices]

    # 5. Filter the list to get the sequence pattern (removing identical consecutive frames)
    pattern = []
    if raw_notes:
        pattern.append(raw_notes[0])
        for i in range(1, len(raw_notes)):
            if raw_notes[i] != raw_notes[i-1]:
                pattern.append(raw_notes[i])

    logging.info(f"Note Pattern: {pattern}")
    return pattern

def identify_notes(file_path):
    # 1. Load the WAV file
    y, sr = librosa.load(file_path, sr=None)

    # 2. Extract Harmonic Component (Ignore percussion/harmonics)
    y_harmonic = librosa.effects.harmonic(y)

    # 3. Compute Chromagram (Note Pattern)
    # Chroma represents the 12 semitones of the musical octave
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # 4. Identify dominant note over time
    # Sum over time or find maximum in each frame
    note_pattern = np.argmax(chroma, axis=0)

    # Convert note numbers to note names
    raw_notes = [OCTAVE_NOTES[note] for note in note_pattern]

    # 5. Filter the list to get the sequence pattern (removing identical consecutive frames)
    pattern = []
    if raw_notes:
        pattern.append(raw_notes[0])
        for i in range(1, len(raw_notes)):
            if raw_notes[i] != raw_notes[i-1]:
                pattern.append(raw_notes[i])

    logging.info(f"File: {file_path}; Detected Notes: {pattern}")
    return pattern

def identify_westminster_chimes_shifting_pitch(wav_path):
    logging.info(f"Wave File: {wav_path}")
    logging.info(f"CHANGE_Q4: {CHANGE_Q4}")
    # 1. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(wav_path, sr=None)

    # 2. Shift the pitch by 2 semitones
    # n_steps can be fractional (e.g., 2.5) for fine-tuning
    notes = identify_westminster_chimes(y, sr)
    # playback_notes(notes)

def identify_westminster_chimes(y, sr):
    # The "ring" is a complex combination of frequencies that decay over time.
    # The "strike note" (often an octave above the nominal pitch) is critical for identification

    # 2. Extract Harmonic and Percussive components
    # This helps ignore harmonic overtones and focus on the note's fundamental pitch
    # Get a more isolated percussive component by widening its margin
    # Returns two time series, containing the harmonic (tonal) and percussive (transient) portions of the signal.
    # The motivation for this kind of operation is two-fold: first, percussive elements tend to be stronger indicators
    # of rhythmic content, and can help provide more stable beat tracking results; second, percussive elements can
    # pollute tonal feature representations (such as chroma) by contributing energy across all frequency bands,
    # so we’d be better off without them.
    # Bells have a sharp strike (percussive) and a long tone (harmonic).
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))
    #
    # # Beat track on the percussive signal
    # tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr, units='time')
    # logging.info(f"Beat Frames: {beat_frames}")

    # 3. Onset detection: Find when new notes start
    # Locate note onset events by picking peaks in an onset strength envelope.
    # Set the units to encode detected onset events in to time.
    onsets = librosa.onset.onset_detect(y=y_harmonic, sr=sr, units='time')
    # logging.info(f"Onsets: {onsets}")

    # Add start and end points for segmentation
    segment_times = np.concatenate(([0], onsets, [librosa.get_duration(y=y, sr=sr)]))

    detected_notes = []

    # 4. Analyze pitch for each segment
    for i in range(len(segment_times) - 1):
        start = segment_times[i]
        end = segment_times[i + 1]

        # Only analyze segments long enough to be notes
        if end - start < 0.2: continue

        # Extract segment
        segment = y_harmonic[int(start * sr):int(end * sr)]

        # Calculate pitch using Normalized Cross-Correlation (pitch detection)
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)

        # Find the dominant pitch (default 0.5)
        pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.85])

        if not np.isnan(pitch):
            detected_notes.append([float(start), float(end-start), librosa.hz_to_note(pitch)])

    # 5. Map Notes to Westminster Chime Pattern
    # Traditional Westminster chimes: G4, C5, D5, G4 (or equivalent transposed)
    # The pattern often starts with: G4, C5, D5, G4

    # Clean up notes (approximate to nearest, ignore octaves if necessary)
    # For this example, we look for the sequence pattern.

    detected_notes_compressed = []
    if detected_notes:
        detected_notes_compressed.append(detected_notes[0])
        for i in range(1, len(detected_notes)):
            if detected_notes[i-1][2] == detected_notes[i][2]:
                # Add the duration to the last entry and throw this one away...
                detected_notes_compressed[-1][1] = detected_notes_compressed[-1][1] + detected_notes[i][1]
            else:
                # It's different so add it to the pattern...
                detected_notes_compressed.append(detected_notes[i])

    # detected_notes_loudness = get_loudest_notes(y, sr)

    print_str = ", ".join(f"{i[2]} {i[0]:.3f}:{i[1]:.3f}" for i in detected_notes_compressed)
    logging.info(f"Detected Notes: {print_str}")

    # Simple pattern recognition
    pattern = ["G4", "C5", "D5", "G4"]
    # This logic requires refinement based on the specific transposition of your wav
    if any(p in " ".join([i[2] for i in detected_notes_compressed]) for p in CHANGE_Q4):
        logging.info(f"Pattern: Westminster Chimes")
    else:
        logging.info(f"Pattern: Not Found")

    return detected_notes_compressed

def playback_notes(notes):
    sr = 22050
    interval = 0.5
    # Synthesize and play notes
    for note in notes:
        # 1. Convert note name to frequency
        freq = librosa.note_to_hz(note[2])

        # 2. Generate tone
        note_wave = librosa.tone(freq, sr=sr, duration=note[1])

        # 3. Add silent padding to create the interval
        silence = np.zeros(int(sr * interval))
        audio_with_pause = np.concatenate([note_wave, silence])

        # 4. Play note
        print(f"Playing {note}...")
        sd.play(note_wave, sr)
        sd.wait()  # Wait for note to finish

def get_loudest_notes(y, sr, top_n=12):
    # 2. Compute Constant-Q Transform (CQT)
    # CQT is ideal for music because bins correspond to musical notes
    cqt = np.abs(librosa.cqt(y, sr=sr))

    # 3. Average the magnitude over the entire duration (time axis)
    # This gives us the 'strength' of each note across the whole track
    mean_cqt = np.mean(cqt, axis=1)

    # 4. Find indices of the highest magnitudes
    top_indices = np.argsort(mean_cqt)[-top_n:][::-1]

    # 5. Convert bin indices to note names (e.g., 'C4', 'A#3')
    # librosa.cqt_frequencies gives the center frequency of each bin
    freqs = librosa.cqt_frequencies(n_bins=len(mean_cqt), fmin=librosa.note_to_hz('C5'))

    loudest_notes = []
    for idx in top_indices:
        note_name = librosa.hz_to_note(freqs[idx])
        magnitude = mean_cqt[idx]
        loudest_notes.append((note_name, magnitude))

    loudest_notes_str = ", ".join(f"{ln[0]} {float(ln[1]):.3f}" for ln in loudest_notes)
    logging.info(f"Loudest Notes: {loudest_notes_str}")
    return loudest_notes

def plot_westminster_chimes(filename):
    y, sr = librosa.load(filename)

    # 2. Harmonic-Percussive Source Separation (HPSS)
    # Bells have a sharp strike (percussive) and a long tone (harmonic).
    # Separating them helps focus on the fundamental frequency.
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # 3. Melody/Pitch Extraction using pYIN
    # pYIN is effective for tracking musical notes (F0).
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_harmonic,
        fmin = librosa.note_to_hz('C4'),  # Low bell range
        fmax = librosa.note_to_hz('C6'),  # High bell range
        sr=sr
    )

    # 4. Convert Pitch (Hz) to Notes
    # The Westminster Chime often uses B3, E4, D4, C4, G4, etc.
    times = librosa.times_like(f0)
    clean_f0 = np.nan_to_num(f0, nan=0.1)
    notes = librosa.hz_to_note(clean_f0)

    # Filter out unvoiced frames (no note detected)
    non_nan_indices = ~np.isnan(f0)
    detected_times = times[non_nan_indices]
    detected_notes = notes[non_nan_indices]

    print("Detected notes sequence:")
    # Print the first few notes for brevity
    print(detected_notes[:20])

    # 5. Optional: Visualize the pitch tracking
    plt.figure(figsize=(10, 4))
    plt.plot(times, f0, label='Fundamental Frequency (F0)')
    plt.title('Westminster Chime Melody Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    logging.info("Starting...")
    p = pyaudio.PyAudio()
    parser = argparse.ArgumentParser(description="Westminster argument parser.")
    parser.add_argument("-d", "--devices", action="store_true",
                        help="Print device information before starting.")
    args = parser.parse_args()
    if args.devices:
        for i in range(p.get_device_count()):
            logging.info(p.get_device_info_by_index(i))
    # listen_westminster(p)
    # listen_for_peaks(p, 40, './logs/output.wav')
    # https://medium.com/@ianvonseggern/note-recognition-in-python-c2020d0dae24

    # listen_for_peaks_in_file(p,'./chime_audio/ChristChurchCathedralDublin_20251204.wav')

    # https://sound-effects.bbcrewind.co.uk/search?q=Big%20Ben
    # listen_for_peaks_in_file(p,'./chime_audio/bbc_big_ben_07002151.wav')

    # identify_note_pattern('./chime_audio/bbc_big_ben_07002151.wav')

    # identify_notes('./chime_audio/bbc_big_ben_07002151.wav')
    # identify_notes('./chime_audio/ChristChurchCathedralDublin_20251204.wav')

    identify_westminster_chimes_shifting_pitch('./chime_audio/bbc_big_ben_07002151.wav')
    identify_westminster_chimes_shifting_pitch('./chime_audio/ChristChurchCathedralDublin_20251204.wav')

    # plot_westminster_chimes('./chime_audio/bbc_big_ben_07002151.wav')
    # plot_westminster_chimes('./chime_audio/ChristChurchCathedralDublin_20251204.wav')
