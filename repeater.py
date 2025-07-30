import pyaudio
import numpy as np
import time
import wave
import threading
import subprocess
import os
import logging
import random
from dtmf import detect_dtmf
from pydub import AudioSegment
from pydub.playback import play
from threading import Event

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "audio/system",
        "audio/memes", 
        "audio/temp",
        "data/cache",
        "data/context", 
        "data/databases",
        "config/settings",
        "config/prompts",
        "flags"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

class RepeaterConfig:
    """Configuration class for repeater settings"""
    # Audio Settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    THRESHOLD = 500
    SILENCE_TIME = 0.5
    AUDIO_BOOST = 5.0
    
    # Flag paths
    CANCEL_FILE = "flags/cancel_ai.flag"
    AI_READY_FILE = "flags/ai_ready.flag"
    
    # Audio File Paths
    LOADING_FILE = "audio/system/loading_loop.wav"
    DING_FILE = "audio/system/ding.wav"
    TIMEOUT_FILE = "audio/system/timeout.wav"
    AI_MODE_FILE = "audio/system/ai_mode.wav"
    MENU_FILE = "audio/system/menu.wav"
    REPEATER_INFO_FILE = "audio/system/repeater_info.wav"
    
    # CW/Morse Settings
    TONE_FREQ = 800
    TONE_DURATION = 0.2
    TONE_VOLUME = 1.0
    CALLSIGN = "SV2TMT"
    CW_WPM = 20
    CW_FARNSWORTH_WPM = None
    CW_ID_INTERVAL = 600  # seconds
    
    # Morse code dictionary
    MORSE_DICT = {
        'A': ".-", 'B': "-...", 'C': "-.-.", 'D': "-..", 'E': ".", 'F': "..-.", 
        'G': "--.", 'H': "....", 'I': "..", 'J': ".---", 'K': "-.-", 'L': ".-..", 
        'M': "--", 'N': "-.", 'O': "---", 'P': ".--.", 'Q': "--.-", 'R': ".-.", 
        'S': "...", 'T': "-", 'U': "..-", 'V': "...-", 'W': ".--", 'X': "-..-",
        'Y': "-.--", 'Z': "--..", '1': ".----", '2': "..---", '3': "...--", 
        '4': "....-", '5': ".....", '6': "-....", '7': "--...", '8': "---..", 
        '9': "----.", '0': "-----"
    }
    
    # Weather config
    OPENWEATHER_API_KEY = "your_api_key_here"    
    WEATHER_CACHE_FILE = "data/cache/weather_cache.json"
    WEATHER_CACHE_DURATION = 900  # seconds
    
    # Satellite TLEs
    TLE_URLS = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=amateur&FORMAT=tle"
    ]

    TLE_CACHE_FILE = "data/cache/tle_cache.txt"
    SATELLITES_TO_TRACK = ["ISS (ZARYA)", "NOAA 15", "NOAA 19", "RADFXSAT (FOX-1B)", "SAUDISAT 1C (SO-50)"]
    
    # Callsign database
    RADIOID_CSV_URL = "https://radioid.net/static/user.csv"
    RADIOID_LOCAL_FILE = "data/cache/user.csv"
    
    PHONETIC_MAP = {
    # English
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
    "golf": "G", "hotel": "H", "india": "I", "juliett": "J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R",
    "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
    "x-ray": "X", "xray": "X", "yankee": "Y", "zulu": "Z",

    # Greek/phonetic variants
    "σιέρα": "S", "βίκτορ": "V", "μάικ": "M", "τάγκο": "T", "λίμα": "L", "νοβέμπερ": "N",
    "όσκαρ": "O", "πάπα": "P", "κίλο": "K", "τζούλιετ": "J", "γκόλφ": "G", "έκο": "E",
    "ντέλτα": "D", "γιάνκι": "Y", "ζουλού": "Z"
    }

    GREEK_NUMBER_MAP = {
    "μηδέν": "0", "ένα": "1", "δύο": "2", "τρία": "3", "τέσσερα": "4",
    "πέντε": "5", "έξι": "6", "επτά": "7", "οκτώ": "8", "εννέα": "9",
    "μία": "1", "δυο": "2", "τρεις": "3", "τεσσερα": "4", "εννιά": "9"  # alternate spellings
    }
    
    ENGLISH_NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }

class HamRepeater:
    def __init__(self):
        self.config = RepeaterConfig()
        self.dt_detected = None
        self.morse_wpm = self.config.CW_WPM
        self.morse_farnsworth_wpm = self.config.CW_FARNSWORTH_WPM
        self.play_ai_mode = False
        self.ai_mode_running = False
        self.talking = False
        self.play_menu = False
        self.play_info = False
        self.play_meme = False
        self.play_fact = False
        self.play_time = False
        self.play_weather = False
        self.play_band = False
        self.play_satpass = False
        self.lookup_callsign = False
        self.callsign_data = self.load_radioid_data()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self._setup_audio_streams()
        
    def _setup_audio_streams(self):
        """Initialize audio input and output streams"""
        try:
            self.input_stream = self.p.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            self.output_stream = self.p.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                output=True
            )
            logger.info("Audio streams initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio streams: {e}")
            raise
    
    def play_audio(self, filename):
        """Play a WAV file through the audio output"""
        try:
            with wave.open(filename, 'rb') as wf:
                audio_stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                data = wf.readframes(self.config.CHUNK)
                while data:
                    audio_stream.write(data)
                    data = wf.readframes(self.config.CHUNK)
                
                audio_stream.close()
                
        except FileNotFoundError:
            logger.error(f"Audio file {filename} not found!")
        except Exception as e:
            logger.error(f"Error playing audio file {filename}: {e}")
    
    def calculate_timing(self, wpm, farnsworth_wpm=None):
        """
        Calculate morse code timing based on WPM and Farnsworth timing
        
        Standard timing:
        - Dit duration = 1.2 / WPM seconds
        - Dah duration = 3 × dit duration
        - Inter-element gap = 1 × dit duration
        - Inter-character gap = 3 × dit duration
        - Inter-word gap = 7 × dit duration
        
        Farnsworth timing:
        - Character timing based on character WPM
        - Spacing timing based on effective WPM (slower)
        """
        # Calculate dit duration based on character speed
        dit_duration = 1.2 / wpm
        
        # If Farnsworth timing is used, calculate slower spacing
        if farnsworth_wpm and farnsworth_wpm < wpm:
            # Character elements use faster timing
            char_dit_duration = dit_duration
            # Spacing uses slower timing
            spacing_dit_duration = 1.2 / farnsworth_wpm
            
            # Calculate how much extra time to add to spacing
            extra_spacing_time = spacing_dit_duration - char_dit_duration
            
            return {
                'dit_duration': char_dit_duration,
                'dah_duration': char_dit_duration * 3,
                'element_gap': char_dit_duration,
                'inter_character_gap': (char_dit_duration * 3) + (extra_spacing_time * 4),
                'inter_word_gap': (char_dit_duration * 7) + (extra_spacing_time * 12)
            }
        else:
            # Standard timing
            return {
                'dit_duration': dit_duration,
                'dah_duration': dit_duration * 3,
                'element_gap': dit_duration,
                'inter_character_gap': dit_duration * 3,
                'inter_word_gap': dit_duration * 7
            }

    def set_morse_wpm(self, wpm, farnsworth_wpm=None):
        """Set the WPM and optional Farnsworth WPM"""
        self.morse_wpm = wpm
        self.morse_farnsworth_wpm = farnsworth_wpm
        logger.info(f"Set morse WPM to {wpm}" + 
                   (f" with Farnsworth timing at {farnsworth_wpm} WPM" if farnsworth_wpm else ""))

    def play_tone(self, frequency=None, duration=None, volume=None):
        """Generate and play a tone with proper buffering"""
        frequency = frequency or self.config.TONE_FREQ
        duration = duration or self.config.TONE_DURATION
        volume = volume or self.config.TONE_VOLUME
        
        try:
            # Calculate number of samples
            num_samples = int(self.config.RATE * duration)
            
            # Generate samples with smooth fade in/out to prevent clicks
            samples = np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.config.RATE)
            
            # Apply fade in/out to prevent audio clicks (5ms fade)
            fade_samples = int(0.005 * self.config.RATE)
            if num_samples > 2 * fade_samples:
                # Fade in
                samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out  
                samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            samples = (volume * samples).astype(np.float32)
            
            # Use larger buffer size to prevent underruns
            chunk_size = 1024
            
            stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.RATE,
                output=True,
                frames_per_buffer=chunk_size
            )
            
            # Write audio in chunks to prevent buffer underruns
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                # Pad last chunk if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                stream.write(chunk.tobytes())
            
            # Wait for audio to finish playing
            time.sleep(duration)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error playing tone: {e}")

    def morse_code_tone(self, character, timing):
        """Play morse code for a single character with calculated timing"""
        morse_pattern = self.config.MORSE_DICT.get(character.upper(), '')
        
        if not morse_pattern:
            return
        
        for i, symbol in enumerate(morse_pattern):
            if symbol == '.':
                self.play_tone(duration=timing['dit_duration'])
            elif symbol == '-':
                self.play_tone(duration=timing['dah_duration'])
            
            # Add gap between elements within the same character (except after last element)
            if i < len(morse_pattern) - 1:
                time.sleep(timing['element_gap'])

    def play_callsign_morse(self, wpm=None, farnsworth_wpm=None):
        """Optimized version that generates entire callsign audio at once"""
        # Use provided WPM or fall back to instance settings or default
        current_wpm = wpm or getattr(self, 'morse_wpm', 20)
        current_farnsworth = farnsworth_wpm or getattr(self, 'morse_farnsworth_wpm', None)
        
        # Calculate timing based on WPM settings
        timing = self.calculate_timing(current_wpm, current_farnsworth)
        
        timing_info = f"WPM: {current_wpm}"
        if current_farnsworth:
            timing_info += f", Farnsworth: {current_farnsworth}"
        
        logger.info(f"Transmitting callsign {self.config.CALLSIGN} in CW ({timing_info})")
        
        frequency = self.config.TONE_FREQ
        volume = self.config.TONE_VOLUME
        rate = self.config.RATE
        
        # Build complete audio signal
        audio_data = []
        
        for i, char in enumerate(self.config.CALLSIGN):
            if char.isalnum():
                morse_pattern = self.config.MORSE_DICT.get(char.upper(), '')
                
                for j, symbol in enumerate(morse_pattern):
                    # Generate tone
                    duration = timing['dit_duration'] if symbol == '.' else timing['dah_duration']
                    num_samples = int(rate * duration)
                    samples = np.sin(2 * np.pi * np.arange(num_samples) * frequency / rate)
                    
                    # Apply fade to prevent clicks
                    fade_samples = int(0.005 * rate)
                    if num_samples > 2 * fade_samples:
                        samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
                        samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                    
                    audio_data.extend(volume * samples)
                    
                    # Add gap between elements
                    if j < len(morse_pattern) - 1:
                        gap_samples = int(rate * timing['element_gap'])
                        audio_data.extend(np.zeros(gap_samples))
                
                # Add inter-character gap
                if i < len(self.config.CALLSIGN) - 1:
                    next_char = self.config.CALLSIGN[i + 1] if i + 1 < len(self.config.CALLSIGN) else ''
                    if next_char.isalnum():
                        gap_samples = int(rate * timing['inter_character_gap'])
                        audio_data.extend(np.zeros(gap_samples))
            elif char == ' ':
                # Add word gap
                gap_samples = int(rate * timing['inter_word_gap'])
                audio_data.extend(np.zeros(gap_samples))
        
        # Convert to numpy array and play
        if audio_data:
            audio_array = np.array(audio_data, dtype=np.float32)
            
            try:
                stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=rate,
                    output=True,
                    frames_per_buffer=1024
                )
                
                # Play the entire audio buffer
                chunk_size = 1024
                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i+chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    stream.write(chunk.tobytes())
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Error playing morse code: {e}")

    def play_text_morse(self, text, wpm=None, farnsworth_wpm=None):
        """Play any text in morse code"""
        # Use provided WPM or fall back to instance settings or default
        current_wpm = wpm or getattr(self, 'morse_wpm', 20)
        current_farnsworth = farnsworth_wpm or getattr(self, 'morse_farnsworth_wpm', None)
        
        # Calculate timing based on WPM settings
        timing = self.calculate_timing(current_wpm, current_farnsworth)
        
        timing_info = f"WPM: {current_wpm}"
        if current_farnsworth:
            timing_info += f", Farnsworth: {current_farnsworth}"
        
        logger.info(f"Transmitting '{text}' in CW ({timing_info})")
        
        for i, char in enumerate(text):
            if char.isalnum():
                self.morse_code_tone(char, timing)
                
                # Add inter-character gap (except after last character)
                if i < len(text) - 1:
                    next_char = text[i + 1] if i + 1 < len(text) else ''
                    if next_char.isalnum():
                        time.sleep(timing['inter_character_gap'])
            elif char == ' ':
                # Handle word spacing
                time.sleep(timing['inter_word_gap'])
    
    def play_looping_audio(self, filename, stop_event):
        """Play audio file in a loop until stop event is set"""
        try:
            sound = AudioSegment.from_wav(filename)
            while not stop_event.is_set():
                play(sound)
        except Exception as e:
            logger.error(f"Error in looping audio: {e}")
    
    def cleanup_files(self):
        """Remove temporary flag files"""
        files_to_remove = [self.config.CANCEL_FILE, self.config.AI_READY_FILE]
        for file in files_to_remove:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except OSError as e:
                    logger.warning(f"Could not remove {file}: {e}")
    
    def run_ai_mode(self):
        """Run AI mode subprocess with proper monitoring"""
        self.ai_mode_running = True
        timeout_played = [False]  # Use list to make it mutable in nested function
        ai_ready = [False]
        
        logger.info("Starting AI mode")
        self.cleanup_files()
        
        # Start loading audio loop
        loading_stop_event = Event()
        loading_thread = threading.Thread(
            target=self.play_looping_audio, 
            args=(self.config.LOADING_FILE, loading_stop_event),
            daemon=True
        )
        loading_thread.start()
        
        # Start AI subprocess
        try:
            process = subprocess.Popen(["python", "aimode.py"])
        except Exception as e:
            logger.error(f"Failed to start AI mode subprocess: {e}")
            loading_stop_event.set()
            loading_thread.join()
            self.ai_mode_running = False
            return
        
        def monitor_ai_ready():
            while process.poll() is None:  # While process is running
                if os.path.exists(self.config.AI_READY_FILE):
                    loading_stop_event.set()
                    loading_thread.join()
                    self.play_audio(self.config.DING_FILE)
                    ai_ready[0] = True
                    logger.info("AI mode ready")
                    return
                time.sleep(0.1)
            
            # Process exited before ready signal - early exit
            loading_stop_event.set()
            loading_thread.join()
            self.play_audio(self.config.TIMEOUT_FILE)
            timeout_played[0] = True
            logger.warning("AI mode exited before ready signal")
        
        monitor_thread = threading.Thread(target=monitor_ai_ready, daemon=True)
        monitor_thread.start()
        
        # Wait for AI process to complete
        process.wait()
        monitor_thread.join(timeout=1.0)  # Give monitor thread time to finish
        
        # Cleanup
        self.cleanup_files()
        
        # Only play timeout if AI became ready but then exited normally
        # Don't play if it was an early exit (already played) or if still loading
        if ai_ready[0] and not timeout_played[0]:
            time.sleep(0.6)
            self.play_audio(self.config.TIMEOUT_FILE)
        
        self.ai_mode_running = False
        logger.info("AI mode completed")
        
    def load_radioid_data(self):
        import csv
        import requests
        import sqlite3
        
        # Create/update SQLite database instead of loading everything into memory
        db_file = "data/databases/callsigns.db"
        
        try:
            # Download fresh data if needed
            response = requests.get(self.config.RADIOID_CSV_URL, timeout=10)
            if response.status_code == 200:
                with open(self.config.RADIOID_LOCAL_FILE, "w", encoding="utf-8") as f:
                    f.write(response.text)
                logger.info("Downloaded fresh user.csv from RadioID")
                
                # Convert CSV to SQLite for faster lookups
                conn = sqlite3.connect(db_file)
                conn.execute('''CREATE TABLE IF NOT EXISTS callsigns 
                               (callsign TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, 
                                city TEXT, country TEXT)''')
                
                with open(self.config.RADIOID_LOCAL_FILE, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        conn.execute('''INSERT OR REPLACE INTO callsigns 
                                       VALUES (?, ?, ?, ?, ?)''',
                                   (row["CALLSIGN"].strip().upper(),
                                    row["FIRST_NAME"].strip(),
                                    row["LAST_NAME"].strip(),
                                    row["CITY"].strip(),
                                    row["COUNTRY"].strip()))
                
                conn.commit()
                conn.close()
                logger.info("Converted CSV to SQLite database")
                
        except Exception as e:
            logger.warning(f"Could not download/process RadioID CSV: {e}")
        
        return db_file  # Return database filename instead of records dict
        
    def lookup_callsign_in_db(self, callsign):
        """Look up callsign in SQLite database"""
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.callsign_data)  # self.callsign_data now contains DB filename
            cursor = conn.execute('SELECT * FROM callsigns WHERE callsign = ?', (callsign.upper(),))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "first": result[1],
                    "last": result[2], 
                    "city": result[3],
                    "country": result[4]
                }
            return None
        except Exception as e:
            logger.error(f"Database lookup error: {e}")
            return None
    
    def dtmf_listener(self):
        """Listen for DTMF tones and handle commands"""
        while True:
            try:
                detected = detect_dtmf()
                if detected:
                    self.dt_detected = detected
                    logger.info(f"DTMF detected: {detected}")
                    
                    if detected == '*':
                        if self.ai_mode_running:
                            # Cancel AI mode
                            try:
                                with open(self.config.CANCEL_FILE, "w") as f:
                                    f.write("cancel")
                                logger.info("AI mode cancel signal sent")
                            except Exception as e:
                                logger.error(f"Failed to write cancel file: {e}")
                        else:
                            # Start AI mode
                            self.play_ai_mode = True
                            logger.info("AI mode activation requested")
                            
                    elif detected == '#':
                        if not self.play_menu:
                            self.play_menu = True
                            logger.info("Menu playback scheduled")
                            
                    elif detected == '0':
                        if not self.play_info:
                            self.play_info = True
                            logger.info("Repeater info playback scheduled")
                            
                    elif detected == '7':
                        if not self.play_meme:
                            self.play_meme = True
                            logger.info("Random meme playback scheduled")
                            
                    elif detected == '4':
                        if not self.play_fact:
                            self.play_fact = True
                            logger.info("Random fun fact playback scheduled")
                            
                    elif detected == '1':
                        if not self.play_time:
                            self.play_time = True
                            logger.info("Time and date playback scheduled")
                            
                    elif detected == '2':
                        if not self.play_weather:
                            self.play_weather = True
                            logger.info("Weather playback scheduled")
                            
                    elif detected == '3':
                        if not self.play_band:
                            self.play_band = True
                            logger.info("Band conditions playback scheduled")
                            
                    elif detected == '6':
                        if not self.play_satpass:
                            self.play_satpass = True
                            logger.info("Satellite pass playback scheduled")
                            
                    elif detected == '5':
                        if not self.lookup_callsign:
                            self.lookup_callsign = True
                            logger.info("Callsign lookup scheduled")
                            
            except Exception as e:
                logger.error(f"Error in DTMF listener: {e}")
                time.sleep(1)  # Brief pause before retrying
    
    def callsign_thread(self):
        """Periodic callsign identification thread"""
        while True:
            time.sleep(self.config.CW_ID_INTERVAL)
            # Wait if currently talking
            while self.talking:
                time.sleep(1)
            self.play_callsign_morse()
            
    def fetch_tles(self):
        combined = ""
        try:
            import requests
            from pathlib import Path
        
            for url in self.config.TLE_URLS:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                combined += response.text + "\n"
            with open(self.config.TLE_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(combined)
            return combined
        except Exception as e:
            logger.warning(f"Could not fetch TLEs, using cache: {e}")
            if Path(self.config.TLE_CACHE_FILE).exists():
                return Path(self.config.TLE_CACHE_FILE).read_text(encoding="utf-8")
            raise RuntimeError("No valid TLE data available.")
                
    def record_callsign_audio(self, duration=10, filename="audio/temp/callsign.wav"):
        try:
            stream = self.p.open(
                format=self.config.FORMAT,
                channels=1,
                rate=self.config.RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK
            )
            logger.info("Recording callsign...")
            frames = []

            for _ in range(0, int(self.config.RATE / self.config.CHUNK * duration)):
                data = stream.read(self.config.CHUNK, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            wf = wave.open(filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(self.config.FORMAT))
            wf.setframerate(self.config.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            logger.info("Recording complete.")
            return filename
        except Exception as e:
            logger.error(f"Failed to record callsign: {e}")
            return None
            
    def transcribe_audio_whisper(self, filename):
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("small", compute_type="int8")  # You can change model size
            segments, _ = model.transcribe(filename, language="en")
            result = "".join([seg.text for seg in segments]).strip().upper()
            logger.info(f"Transcribed callsign input: {result}")
            return result
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""
            
    def extract_callsign_from_text(self, text):
        words = text.strip().lower().split()
        callsign = ""
        
        # More aggressive callsign extraction
        for word in words:
            # Direct phonetic alphabet mapping
            if word in self.config.PHONETIC_MAP:
                callsign += self.config.PHONETIC_MAP[word]
            elif word in self.config.GREEK_NUMBER_MAP:
                callsign += self.config.GREEK_NUMBER_MAP[word]
            elif word in self.config.ENGLISH_NUMBER_MAP:
                callsign += self.config.ENGLISH_NUMBER_MAP[word]
            elif word.isdigit():
                callsign += word
            elif len(word) == 1 and word.isalpha():
                callsign += word.upper()
            else:
                # Try partial matches for common callsign patterns
                for phonetic, letter in self.config.PHONETIC_MAP.items():
                    if phonetic in word or word in phonetic:
                        callsign += letter
                        break
                else:
                    # Last resort: take first letter if it looks like speech
                    if word.isalpha() and len(word) > 1:
                        callsign += word[0].upper()
        
        # Clean up common patterns and validate
        callsign = callsign.replace(" ", "").upper()
        
        # Basic callsign validation - should have letters and possibly numbers
        if len(callsign) >= 3 and any(c.isalpha() for c in callsign):
            logger.info(f"Extracted callsign from text '{text}': {callsign}")
            return callsign
        else:
            logger.warning(f"Invalid callsign extracted: {callsign}")
            return ""
    
    def run_repeater(self):
        """Main repeater loop"""
        logger.info("Starting ham repeater...")
        
        silent_time = 0
        was_talking = False
        
        # Start background threads
        dtmf_thread = threading.Thread(target=self.dtmf_listener, daemon=True)
        callsign_thread = threading.Thread(target=self.callsign_thread, daemon=True)
        
        dtmf_thread.start()
        callsign_thread.start()
        
        try:
            while True:
                # Read audio input
                data = self.input_stream.read(self.config.CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # Apply audio boost and clipping
                audio_np = np.clip(
                    audio_np * self.config.AUDIO_BOOST, 
                    -32768, 32767
                ).astype(np.int16)
                boosted_data = audio_np.tobytes()
                
                # Check if audio is above threshold (someone talking)
                if np.max(np.abs(audio_np)) > self.config.THRESHOLD:
                    self.output_stream.write(boosted_data)
                    silent_time = 0
                    was_talking = True
                    self.talking = True
                else:
                    # Handle silence after talking
                    if was_talking:
                        if silent_time == 0:
                            start_silent = time.time()
                        silent_time = time.time() - start_silent
                        
                        if silent_time >= self.config.SILENCE_TIME:
                            # Only play roger beep if AI mode is not about to start
                            if not (self.play_ai_mode and not self.ai_mode_running):
                                self.play_tone()  # Roger beep
                            
                            # Menu playback
                            if self.play_menu:
                                self.play_audio(self.config.MENU_FILE)
                                self.play_menu = False
                                
                            # Repeater Info playback
                            if self.play_info:
                                self.play_audio(self.config.REPEATER_INFO_FILE)
                                self.play_info = False
                                
                            # Meme playback
                            if self.play_meme:
                                try:
                                    meme_folder = "audio/memes"
                                    meme_files = [f for f in os.listdir(meme_folder) if f.lower().endswith(".wav")]
                                    if meme_files:
                                        random_file = random.choice(meme_files)
                                        self.play_audio(os.path.join(meme_folder, random_file))
                                        logger.info(f"Played meme: {random_file}")
                                    else:
                                        logger.warning("No meme files found in memes/ folder")
                                except Exception as e:
                                    logger.error(f"Error during meme playback: {e}")
                                finally:
                                    self.play_meme = False
                            
                            def set_greek_voice(engine):
                                voices = engine.getProperty('voices')
                                for voice in voices:
                                    langs = []
                                    try:
                                        langs = [lang.decode('utf-8') for lang in voice.languages if isinstance(lang, bytes)]
                                    except Exception:
                                        pass
                                    if any('el' in lang.lower() for lang in langs) or 'el' in voice.id.lower():
                                        engine.setProperty('voice', voice.id)
                                        logger.info(f"Greek voice selected: {voice.name}")
                                        return
                                logger.warning("No Greek voice found — using default voice.")
                            
                            # Fun Fact playback
                            if self.play_fact:
                                try:
                                    fact_file = "data/databases/fun_facts.txt"
                                    if os.path.exists(fact_file):
                                        with open(fact_file, "r", encoding="utf-8") as f:
                                            facts = [line.strip() for line in f if line.strip()]
                                        if facts:
                                            fact = random.choice(facts)
                                            logger.info(f"Speaking fun fact: {fact}")

                                            subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/temp_fact.wav', fact])
                                            self.play_audio("audio/temp/temp_fact.wav")
                                            os.remove("audio/temp/temp_fact.wav")
                                        else:
                                            logger.warning("fun_facts.txt is empty")
                                    else:
                                        logger.warning("fun_facts.txt not found")
                                except Exception as e:
                                    logger.error(f"Error during fun fact playback: {e}")
                                finally:
                                    self.play_fact = False
                                    
                            # Time and Date playback
                            if self.play_time:
                                try:
                                    from datetime import datetime

                                    GREEK_HOUR_NAMES = {
                                        1: "μία",
                                        2: "δύο",
                                        3: "τρεις",
                                        4: "τέσσερις",
                                        5: "πέντε",
                                        6: "έξι",
                                        7: "επτά",
                                        8: "οκτώ",
                                        9: "εννέα",
                                        10: "δέκα",
                                        11: "έντεκα",
                                        12: "δώδεκα",
                                        0: "δώδεκα"
                                    }

                                    GREEK_MONTHS = {
                                        "January": "Ιανουαρίου",
                                        "February": "Φεβρουαρίου",
                                        "March": "Μαρτίου",
                                        "April": "Απριλίου",
                                        "May": "Μαΐου",
                                        "June": "Ιουνίου",
                                        "July": "Ιουλίου",
                                        "August": "Αυγούστου",
                                        "September": "Σεπτεμβρίου",
                                        "October": "Οκτωβρίου",
                                        "November": "Νοεμβρίου",
                                        "December": "Δεκεμβρίου"
                                    }

                                    now = datetime.now()
                                    hour = now.hour % 12
                                    minute = now.minute
                                    day = now.day
                                    month_en = now.strftime("%B")
                                    month_gr = GREEK_MONTHS.get(month_en, month_en)
                                    year = now.year

                                    hour_word = GREEK_HOUR_NAMES.get(hour, str(hour))
                                    minute_word = f"{minute}" if minute != 0 else "ακριβώς"

                                    if minute != 0:
                                        time_phrase = f"{hour_word} και {minute_word}"
                                    else:
                                        time_phrase = f"{hour_word} ακριβώς"

                                    date_phrase = f"Σήμερα είναι {day} του {month_gr} του {year}."
                                    full_phrase = f"Η ώρα είναι {time_phrase}. {date_phrase}"

                                    logger.info(f"Speaking time and date: {full_phrase}")

                                    subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/phrase.wav', full_phrase])
                                    self.play_audio("audio/temp/phrase.wav")
                                    os.remove("audio/temp/phrase.wav")

                                except Exception as e:
                                    logger.error(f"Error during time and date playback: {e}")
                                finally:
                                    self.play_time = False
                                    
                            # Weather playback
                            if self.play_weather:
                                try:
                                    import requests, json
                                    from datetime import datetime

                                    city = "Serres,GR"
                                    api_key = self.config.OPENWEATHER_API_KEY
                                    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=el"

                                    cache_valid = False
                                    forecast = ""

                                    # Check cache
                                    if os.path.exists(self.config.WEATHER_CACHE_FILE):
                                        with open(self.config.WEATHER_CACHE_FILE, "r", encoding="utf-8") as f:
                                            cache_data = json.load(f)
                                            timestamp = cache_data.get("timestamp", 0)
                                            if time.time() - timestamp < self.config.WEATHER_CACHE_DURATION:
                                                forecast = cache_data.get("forecast", "")
                                                cache_valid = True

                                    # If no valid cache, fetch new data
                                    if not cache_valid:
                                        response = requests.get(url)
                                        if response.status_code == 200:
                                            data = response.json()
                                            temp = round(data["main"]["temp"])
                                            description = data["weather"][0]["description"]
                                            humidity = data["main"]["humidity"]
                                            windspeed = round(data["wind"]["speed"])

                                            forecast = (
                                                f"Ο καιρός στις Σέρρες είναι {description}, "
                                                f"με θερμοκρασία {temp} βαθμούς Κελσίου, "
                                                f"υγρασία {humidity} τοις εκατό,"
                                                f"και ανέμους {windspeed} μέτρα το δευτερόλεπτο."
                                            )

                                            # Save to cache
                                            with open(self.config.WEATHER_CACHE_FILE, "w", encoding="utf-8") as f:
                                                json.dump({"forecast": forecast, "timestamp": time.time()}, f)
                                        else:
                                            logger.warning(f"Failed to fetch weather: {response.status_code}")
                                            forecast = "Δεν μπόρεσα να ανακτήσω τα δεδομένα καιρού."

                                    logger.info(f"Speaking weather: {forecast}")
                                    subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/forecast.wav', forecast])
                                    self.play_audio("audio/temp/forecast.wav")
                                    os.remove("audio/temp/forecast.wav")

                                except Exception as e:
                                    logger.error(f"Error during weather playback: {e}")
                                finally:
                                    self.play_weather = False
                                    
                            # Band Conditions playback
                            if self.play_band:
                                try:
                                    import requests
                                    import xml.etree.ElementTree as ET

                                    response = requests.get("https://www.hamqsl.com/solarxml.php")
                                    if response.status_code == 200:
                                        root = ET.fromstring(response.content)
                                        data = root.find("solardata")

                                        # Basic space weather
                                        sfi = data.findtext("solarflux", default="N/A").strip()
                                        kindex = data.findtext("kindex", default="N/A").strip()
                                        sunspots = data.findtext("sunspots", default="N/A").strip()
                                        xray = data.findtext("xray", default="N/A").strip()
                                        noise = data.findtext("signalnoise", default="N/A").strip()

                                        # HF conditions (daytime)
                                        bands = data.find("calculatedconditions")
                                        band_reports = {}
                                        for band in bands.findall("band"):
                                            if band.attrib.get("time") == "day":
                                                band_reports[band.attrib["name"]] = band.text.strip()

                                        # Build Greek report
                                        band_phrases = []
                                        greek_band_names = {
                                            "80m-40m": "80 και 40 μέτρα",
                                            "30m-20m": "30 και 20 μέτρα",
                                            "17m-15m": "17 και 15 μέτρα",
                                            "12m-10m": "12 και 10 μέτρα"
                                        }

                                        for key, condition in band_reports.items():
                                            greek_band = greek_band_names.get(key, key)
                                            band_phrases.append(f"{greek_band}: {condition}")

                                        band_phrase = ", ".join(band_phrases)
                                        full_report = (
                                            f"Ο δείκτης ηλιακής ροής είναι {sfi}, "
                                            f"ο δείκτης Κ είναι {kindex}, "
                                            f"ο αριθμός των ηλιακών κηλίδων είναι {sunspots}, "
                                            f"και η ακτινοβολία X-ray είναι {xray}. "
                                            f"Θόρυβος σήματος: {noise}. "
                                            f"Καταστάσεις HF κατά τη διάρκεια της ημέρας: {band_phrase}."
                                        )

                                        logger.info(f"Speaking band conditions: {full_report}")
                                        subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/report.wav', full_report])
                                        self.play_audio("audio/temp/report.wav")
                                        os.remove("audio/temp/report.wav")
                                    else:
                                        logger.warning("Failed to fetch band condition XML data")
                                except Exception as e:
                                    logger.error(f"Error during band conditions playback: {e}")
                                finally:
                                    self.play_band = False
                                    
                            # Satellite Pass playback
                            if self.play_satpass:
                                try:
                                    import pyttsx3
                                    from datetime import datetime, timedelta
                                    import pytz
                                    from pytz import timezone
                                    from skyfield.api import EarthSatellite, Topos, load, utc

                                    # Observer: Serres
                                    observer = Topos(latitude_degrees=41.08, longitude_degrees=23.55, elevation_m=50)
                                    local_tz = pytz.timezone("Europe/Athens")
                                    ts = load.timescale()
                                    now = ts.now()
                                    t_end = ts.utc((datetime.utcnow() + timedelta(hours=24)).replace(tzinfo=utc))

                                    # Fetch TLEs
                                    tle_text = self.fetch_tles()
                                    lines = tle_text.strip().splitlines()

                                    # Parse TLEs into Skyfield satellites - FIXED VERSION
                                    satellites = []
                                    
                                    # More precise name matching
                                    def name_matches(tle_name, target_names):
                                        tle_name_upper = tle_name.upper()
                                        for target in target_names:
                                            target_upper = target.upper()
                                            
                                            # Check for exact match first
                                            if target_upper == tle_name_upper:
                                                return True
                                            
                                            # For satellites with numbers, be more precise
                                            if any(char.isdigit() for char in target_upper):
                                                # Extract the base name and number from target
                                                target_parts = target_upper.replace('(', ' ').replace(')', ' ').split()
                                                tle_parts = tle_name_upper.replace('(', ' ').replace(')', ' ').split()
                                                
                                                # Check if all parts of target are in TLE name
                                                if all(part in tle_parts for part in target_parts):
                                                    return True
                                            else:
                                                # For non-numbered satellites, allow partial matching
                                                if target_upper in tle_name_upper or tle_name_upper in target_upper:
                                                    return True
                                        
                                        return False

                                    i = 0
                                    while i < len(lines):
                                        line = lines[i].strip()
                                        
                                        # Skip empty lines
                                        if not line:
                                            i += 1
                                            continue
                                            
                                        # Check if this looks like a satellite name (not starting with 1 or 2)
                                        if not line.startswith(('1 ', '2 ')):
                                            # This should be a satellite name
                                            if i + 2 < len(lines):  # Make sure we have 2 more lines
                                                name = line
                                                line1 = lines[i + 1].strip()
                                                line2 = lines[i + 2].strip()
                                                
                                                # Verify the next two lines are TLE data
                                                if line1.startswith('1 ') and line2.startswith('2 '):
                                                    if name_matches(name, self.config.SATELLITES_TO_TRACK):
                                                        try:
                                                            sat = EarthSatellite(line1, line2, name, ts)
                                                            satellites.append(sat)
                                                            print(f"Added satellite: {name}")  # Debug print
                                                        except Exception as e:
                                                            print(f"Error creating satellite {name}: {e}")
                                                    i += 3  # Skip the 3 lines we just processed
                                                else:
                                                    i += 1  # Just skip this line if it's not followed by TLE data
                                            else:
                                                i += 1
                                        else:
                                            i += 1

                                    print(f"Total satellites loaded: {len(satellites)}")  # Debug print

                                    # Find upcoming AOS events for each satellite
                                    eph = load("de421.bsp")  # Required by skyfield
                                    passes = []
                                    
                                    for sat in satellites:
                                        try:
                                            times, events = sat.find_events(observer, now, t_end, altitude_degrees=10.0)
                                            print(f"Found {len(times)} events for {sat.name}")  # Debug print
                                            
                                            for t, event in zip(times, events):
                                                if event == 0:  # AOS
                                                    dt_utc = t.utc_datetime()
                                                    dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
                                                    passes.append((sat.name, dt_local))
                                                    print(f"AOS for {sat.name}: {dt_local}")  # Debug print
                                                    break  # Only take first AOS per satellite
                                                    
                                        except Exception as e:
                                            print(f"Error finding events for {sat.name}: {e}")

                                    if not passes:
                                        spoken = "Δεν βρέθηκαν επικείμενες διελεύσεις δορυφόρων."
                                    else:
                                        # Sort and limit
                                        passes.sort(key=lambda x: x[1])
                                        spoken_parts = []
                                        greek_hours = {
                                            1: "μία", 2: "δύο", 3: "τρεις", 4: "τέσσερις", 5: "πέντε",
                                            6: "έξι", 7: "επτά", 8: "οκτώ", 9: "εννέα", 10: "δέκα",
                                            11: "έντεκα", 12: "δώδεκα", 0: "δώδεκα"
                                        }

                                        for name, dt in passes[:3]:
                                            hour = dt.hour % 12
                                            minute = dt.minute
                                            hour_str = greek_hours.get(hour, str(hour))
                                            minute_str = f"{minute:02d}"
                                            spoken_parts.append(f"{name}: {hour_str} και {minute_str}")

                                        spoken = "Οι επόμενες διελεύσεις είναι: " + ", ".join(spoken_parts) + " ώρα Ελλάδος."

                                    logger.info(f"Speaking satellite passes: {spoken}")
                                    subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/temp_sat.wav', spoken])
                                    self.play_audio("audio/temp/temp_sat.wav")
                                    os.remove("audio/temp/temp_sat.wav")

                                except Exception as e:
                                    logger.error(f"Error during satellite pass playback: {e}")
                                finally:
                                    self.play_satpass = False
                                    
                            # Callsign lookup
                            if self.lookup_callsign:
                                try:
                                    self.play_audio("audio/system/callsign_prompt.wav")
                                    audio_file = self.record_callsign_audio(duration=8)  # Shorter duration
                                    if not audio_file:
                                        raise Exception("Audio recording failed")

                                    raw_transcription = self.transcribe_audio_whisper(audio_file)
                                    callsign_text = self.extract_callsign_from_text(raw_transcription)
                                    
                                    if not callsign_text:
                                        message = "Δεν κατάλαβα το διακριτικό. Παρακαλώ δοκιμάστε ξανά."
                                    else:
                                        info = self.lookup_callsign_in_db(callsign_text)
                                        if info:
                                            name = f"{info['first']} {info['last']}".strip()
                                            city = info['city']
                                            country = info['country']
                                            message = (
                                                f"Το διακριτικό {' '.join(callsign_text)} ανήκει στον {name} "
                                                f"από {city}, {country}."
                                            )
                                        else:
                                            message = f"Δεν βρέθηκε το διακριτικό {' '.join(callsign_text)}."

                                    logger.info(f"Speaking lookup result: {message}")
                                    subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/message.wav', message])
                                    self.play_audio("audio/temp/message.wav")
                                    os.remove("audio/temp/message.wav")
                                    
                                    # Clean up audio file
                                    if os.path.exists(audio_file):
                                        os.remove(audio_file)

                                except Exception as e:
                                    logger.error(f"Error during callsign lookup: {e}")
                                    # Play error message
                                    error_msg = "Σφάλμα κατά την αναζήτηση διακριτικού."
                                    subprocess.run(['espeak-ng', '-v', 'el', '-s', '145', '-w', 'audio/temp/error.wav', error_msg])
                                    self.play_audio("audio/temp/error.wav")
                                    os.remove("audio/temp/error.wav")
                                finally:
                                    self.lookup_callsign = False

                            was_talking = False
                            self.talking = False
                            silent_time = 0
                            
                            # Check if AI mode should start
                            if self.play_ai_mode and not self.ai_mode_running:
                                self.play_audio(self.config.AI_MODE_FILE)
                                self.play_ai_mode = False
                                ai_thread = threading.Thread(target=self.run_ai_mode, daemon=True)
                                ai_thread.start()
                                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in repeater main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of the repeater"""
        logger.info("Shutting down repeater...")
        
        if self.input_stream:
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.close()
        if self.p:
            self.p.terminate()
        
        self.cleanup_files()
        logger.info("Repeater shutdown complete")

def main():
    """Main entry point"""
    ensure_directories()
    repeater = HamRepeater()
    try:
        repeater.run_repeater()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        repeater.shutdown()

if __name__ == "__main__":
    main()