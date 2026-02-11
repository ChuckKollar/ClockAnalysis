import sounddevice as sd
import numpy as np
import threading
import queue
import sys

# Parameters
SAMPLERATE = 44100  # Standard audio sampling rate
CHANNELS = 1        # Mono audio
DTYPE = 'float32'   # Data type for numpy array
BLOCKSIZE = 1024    # Chunk size for the audio buffer
Q_SIZE = 20         # Max size of the queue

# Create a queue for passing audio data between threads
audio_queue = queue.Queue(maxsize=Q_SIZE)

# The process involves:
# A producer thread (handled automatically by the audio library's callback) that reads the audio and puts it into a queue.
# A consumer thread that gets the audio chunks from the queue and processes them.

# --- Thread 1: Audio Input (Producer) ---
# The callback function runs in a separate thread managed by sounddevice
def callback(indata, frames, time, status):
    """This is called (continuously) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Put the audio data (NumPy array) into the queue
    try:
        audio_queue.put(bytes(indata), block=False)
    except queue.Full:
        # Handle cases where the processing thread is slow
        print("Queue is full, dropping audio block", file=sys.stderr)

# --- Thread 2: Audio Processing (Consumer) ---
def audio_processor():
    """Processes audio chunks from the queue."""
    while True:
        try:
            # Get audio data from the queue
            data_bytes = audio_queue.get(timeout=1)
            # Convert bytes back to a NumPy array for processing
            data_np = np.frombuffer(data_bytes, dtype=DTYPE)

            # !!! Place your audio processing logic here !!!
            
            # Example: simple processing (e.g., calculating the average amplitude)
            amplitude = np.mean(np.abs(data_np))
            print(f"Processing chunk: Mean Amplitude = {amplitude:.4f}")

            # Mark the task as done for the queue
            audio_queue.task_done()

        except queue.Empty:
            # No data in the queue for 1 second, potentially end of stream or program
            print("No audio data in queue, shutting down processor.")
            break
        except Exception as e:
            print(f"An error occurred in the processor thread: {e}")
            break

# --- Main Execution ---
if __name__ == "__main__":
    # Start the processing thread
    processor_thread = threading.Thread(target=audio_processor, daemon=True)
    processor_thread.start()

    print("Starting audio stream and recording (Press Ctrl+C to stop)...")
    try:
        # Open the stream in non-blocking mode with the callback...
        # sounddevice conveniently provides audio data as NumPy arrays, making them easy to
        # process using libraries like NumPy or SciPy. The code converts the bytes back to
        # a NumPy array within the processor thread
        with sd.InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                            channels=CHANNELS, dtype=DTYPE,
                            callback=callback):
            processor_thread.join() # Wait for the processor thread to finish

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"An error occurred with the audio stream: {e}")