# https://github.com/alijamaliz/DTMF-detector modified by Zisis Polychronidis

from scipy.io import wavfile as wav
import pyaudio
import wave
import numpy as np
import time

DTMF_TABLE = {
    '1': [1209, 697], '2': [1336, 697], '3': [1477, 697], 'A': [1633, 697],
    '4': [1209, 770], '5': [1336, 770], '6': [1477, 770], 'B': [1633, 770],
    '7': [1209, 852], '8': [1336, 852], '9': [1477, 852], 'C': [1633, 852],
    '*': [1209, 941], '0': [1336, 941], '#': [1477, 941], 'D': [1633, 941]
} 

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 20000
CHUNK = 1024
RECORD_SECONDS = 0.4

class DTMFDetector:
    def __init__(self, debounce_time=1.0):
        self.last_detected_key = None
        self.last_detection_time = 0
        self.debounce_time = debounce_time
    
    def isNumberInArray(self, array, number, offset=5):
        return any(i in array for i in range(number - offset, number + offset))

    def detect_dtmf(self):
        """Detects DTMF tones and returns the pressed key, or None if no key is detected or if debouncing."""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save and read wave data
        waveFile = wave.open("file.wav", 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
        rate, data = wav.read('file.wav')
        FourierTransformOfData = np.fft.fft(data, 20000)
        FourierTransformOfData = np.abs(FourierTransformOfData).astype(int)

        # Filter and find frequencies
        LowerBound = 20 * np.average(FourierTransformOfData)
        FilteredFrequencies = [i for i in range(len(FourierTransformOfData)) if FourierTransformOfData[i] > LowerBound]

        current_time = time.time()
        
        # Check for DTMF match
        for char, freq_pair in DTMF_TABLE.items():
            if self.isNumberInArray(FilteredFrequencies, freq_pair[0]) and self.isNumberInArray(FilteredFrequencies, freq_pair[1]):
                # Check if this is the same key as last time and within debounce period
                if char == self.last_detected_key and (current_time - self.last_detection_time) < self.debounce_time:
                    return None  # Ignore repeated detection
                
                # Record this detection
                self.last_detected_key = char
                self.last_detection_time = current_time
                return char

        # No tone detected - reset if enough time has passed
        if (current_time - self.last_detection_time) > self.debounce_time:
            self.last_detected_key = None
            
        return None

# Global detector instance for backward compatibility
_global_detector = DTMFDetector()

def detect_dtmf():
    """Backward compatible function that uses a global detector instance."""
    return _global_detector.detect_dtmf()

# Usage examples:
# 
# Option 1 - Using the class directly:
# detector = DTMFDetector(debounce_time=1.0)
# key = detector.detect_dtmf()
#
# Option 2 - Using the standalone function:
# key = detect_dtmf()