import pyaudio
import numpy as np
import time
import wave
import threading
import subprocess
import os
import logging
import random
import argparse
import configparser
import re
import pydub
from logging.handlers import RotatingFileHandler
from pathlib import Path
from piper import PiperVoice
from dtmf import detect_dtmf
from pydub import AudioSegment
from pydub.playback import play
from threading import Event
from collections import deque
from faster_whisper import WhisperModel

# ANSI colors
LOG_COLORS = {
    'DEBUG': '\033[90m',     # Bright Black / Gray
    'INFO': '\033[92m',      # Green
    'WARNING': '\033[93m',   # Yellow
    'ERROR': '\033[91m',     # Red
    'CRITICAL': '\033[95m',  # Magenta
}
RESET_COLOR = '\033[0m'


class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            colored_level = f"{LOG_COLORS[levelname]}{levelname}{RESET_COLOR}"
            record.levelname = colored_level
        return super().format(record)


def setup_logging():
    os.makedirs("logs", exist_ok=True)

    formatter = ColorFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Base logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler (with COLORS!)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File formatter (NO colors for clean logs)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Rotating file log
    file_handler = RotatingFileHandler(
        'logs/repeater.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Error file log
    error_handler = RotatingFileHandler(
        'logs/repeater_errors.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger

logger = setup_logging()

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
        "flags",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

class RepeaterConfig:
    """Configuration class for repeater settings"""
    def __init__(self, args=None, config_file='config/settings/config.ini'):
        self.config = configparser.ConfigParser()
        
        # Check if config file exists
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found! Creating default config...")
            self._create_default_config(config_file)
        
        self.config.read(config_file)
        
        # Audio Settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = self.config.getint('Audio', 'channels', fallback=1)
        self.RATE = self.config.getint('Audio', 'rate', fallback=44100)
        self.CHUNK = self.config.getint('Audio', 'chunk', fallback=1024)
        self.THRESHOLD = self.config.getint('Audio', 'threshold', fallback=500)
        self.SILENCE_TIME = self.config.getfloat('Audio', 'silence_time', fallback=0.5)
        self.AUDIO_BOOST = self.config.getfloat('Audio', 'audio_boost', fallback=5.0)
        self.INPUT_CHANNEL = self.config.get('Audio', 'input_channel', fallback='left').lower()
        self.OUTPUT_VOLUME = self.config.getfloat('Audio', 'output_volume', fallback=1.0)
        
        # Feature flags from arguments
        self.ENABLE_AUDIO_REPEAT = not (args and args.no_audio_repeat)
        self.ENABLE_ROGER_BEEP = not (args and args.no_roger)
        self.ENABLE_CW_ID = not (args and args.no_cw_id)
        self.ENABLE_VOICE_ID = not (args and args.no_voice_id)
        self.ENABLE_DTMF = not (args and args.no_dtmf)
        self.ENABLE_PTT = not (args and args.no_ptt)
        
        # Parse disabled DTMF commands
        self.DISABLED_DTMF_COMMANDS = set()
        if args and args.disable_dtmf:
            # Convert list of command numbers/symbols to a set
            self.DISABLED_DTMF_COMMANDS = set(args.disable_dtmf)
            logger.info(f"Disabled DTMF commands: {', '.join(sorted(self.DISABLED_DTMF_COMMANDS))}")
            
        # DTMF Prefix settings
        self.DTMF_PREFIX = self.config.get('Commands', 'dtmf_prefix', fallback='')
        if self.DTMF_PREFIX:
            logger.info(f"DTMF prefix required: '{self.DTMF_PREFIX}'")
        else:
            logger.info("No DTMF prefix required (direct command mode)")
        
        # Flag paths
        self.CANCEL_FILE = self.config.get('Paths', 'cancel_file', fallback='flags/cancel_ai.flag')
        self.AI_READY_FILE = self.config.get('Paths', 'ai_ready_file', fallback='flags/ai_ready.flag')
        
        # Audio File Paths
        self.LOADING_FILE = self.config.get('Paths', 'loading_file', fallback='audio/system/loading_loop.wav')
        self.DING_FILE = self.config.get('Paths', 'ding_file', fallback='audio/system/ding.wav')
        self.TIMEOUT_FILE = self.config.get('Paths', 'timeout_file', fallback='audio/system/timeout.wav')
        self.AI_MODE_FILE = self.config.get('Paths', 'ai_mode_file', fallback='audio/system/ai_mode.wav')
        self.MENU_FILE = self.config.get('Paths', 'menu_file', fallback='audio/system/menu.wav')
        self.REPEATER_INFO_FILE = self.config.get('Paths', 'repeater_info_file', fallback='audio/system/repeater_info.wav')
        self.CALLSIGN_PROMPT_FILE = self.config.get('Paths', 'callsign_prompt_file', fallback='audio/system/callsign_prompt.wav')
        
        # CW/Morse Settings
        self.TONE_FREQ = self.config.getint('CW', 'tone_freq', fallback=800)
        self.TONE_DURATION = self.config.getfloat('CW', 'tone_duration', fallback=0.2)
        self.TONE_VOLUME = self.config.getfloat('CW', 'tone_volume', fallback=1.0)
        self.CALLSIGN = self.config.get('CW', 'callsign', fallback='SV2TMT')
        self.CW_WPM = self.config.getint('CW', 'wpm', fallback=20)
        farnsworth = self.config.get('CW', 'farnsworth_wpm', fallback='')
        self.CW_FARNSWORTH_WPM = int(farnsworth) if farnsworth.strip() else None
        self.CW_ID_INTERVAL = self.config.getint('CW', 'id_interval', fallback=600)
        
        self.VOICE_ID_INTERVAL = self.config.getint('CW', 'voice_id_interval', fallback=1200)
        
        # Command rate limiting (quota-based)
        self.MAX_COMMANDS = self.config.getint('Commands', 'max_commands', fallback=4)
        self.COMMAND_WINDOW = self.config.getint('Commands', 'window_seconds', fallback=60)
        
        # Morse code dictionary (hardcoded)
        self.MORSE_DICT = {
            'A': ".-", 'B': "-...", 'C': "-.-.", 'D': "-..", 'E': ".", 'F': "..-.", 
            'G': "--.", 'H': "....", 'I': "..", 'J': ".---", 'K': "-.-", 'L': ".-..", 
            'M': "--", 'N': "-.", 'O': "---", 'P': ".--.", 'Q': "--.-", 'R': ".-.", 
            'S': "...", 'T': "-", 'U': "..-", 'V': "...-", 'W': ".--", 'X': "-..-",
            'Y': "-.--", 'Z': "--..", '1': ".----", '2': "..---", '3': "...--", 
            '4': "....-", '5': ".....", '6': "-....", '7': "--...", '8': "---..", 
            '9': "----.", '0': "-----"
        }
        
        # Greek hour names (hardcoded)
        self.GREEK_HOUR_NAMES = {
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
        
        # Piper TTS Settings
        self.PIPER_MODEL_PATH = self.config.get('Piper', 'model_path', fallback='models/el_GR-rapunzelina-low.onnx')
        self.PIPER_TEMP_AUDIO = self.config.get('Piper', 'temp_audio', fallback='audio/temp/piper_temp.wav')
        
        # Weather config
        self.OPENWEATHER_API_KEY = self.config.get('Weather', 'api_key', fallback='your_api_key_here')
        self.WEATHER_CITY = self.config.get('Weather', 'city', fallback='Serres,GR')
        self.WEATHER_CACHE_FILE = self.config.get('Weather', 'cache_file', fallback='data/cache/weather_cache.json')
        self.WEATHER_CACHE_DURATION = self.config.getint('Weather', 'cache_duration', fallback=900)
        
        # Satellite TLEs
        self.TLE_CACHE_FILE = self.config.get('Satellite', 'tle_cache_file', fallback='data/cache/tle_cache.txt')
        
        # Read TLE URLs dynamically
        self.TLE_URLS = []
        i = 1
        while self.config.has_option('Satellite', f'tle_url{i}'):
            self.TLE_URLS.append(self.config.get('Satellite', f'tle_url{i}'))
            i += 1
        
        # Fallback if no URLs found
        if not self.TLE_URLS:
            self.TLE_URLS = [
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=amateur&FORMAT=tle"
            ]
        
        # Satellites to track
        satellites_str = self.config.get('Satellite', 'satellites', 
                                        fallback='ISS (ZARYA), NOAA 15, NOAA 19, RADFXSAT (FOX-1B), SAUDISAT 1C (SO-50)')
        self.SATELLITES_TO_TRACK = [s.strip() for s in satellites_str.split(',')]
        
        # Callsign database
        self.RADIOID_CSV_URL = self.config.get('Callsign', 'radioid_csv_url', 
                                               fallback='https://radioid.net/static/user.csv')
        self.RADIOID_LOCAL_FILE = self.config.get('Callsign', 'radioid_local_file', 
                                                  fallback='data/cache/user.csv')
        self.CALLSIGN_DB_FILE = self.config.get('Callsign', 'database_file',
                                                fallback='data/databases/callsigns.db')
        
        # Location settings
        self.LATITUDE = self.config.getfloat('Location', 'latitude', fallback=41.08)
        self.LONGITUDE = self.config.getfloat('Location', 'longitude', fallback=23.55)
        self.ELEVATION = self.config.getfloat('Location', 'elevation', fallback=50)
        self.TIMEZONE = self.config.get('Location', 'timezone', fallback='Europe/Athens')
        
        # Database paths
        self.FUN_FACTS_FILE = self.config.get('Database', 'fun_facts_file',
                                             fallback='data/databases/fun_facts.txt')
        
        # Phonetic and number maps (hardcoded)
        self.PHONETIC_MAP = {
            # English
            "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
            "golf": "G", "hotel": "H", "india": "I", "juliett": "J", "kilo": "K", "lima": "L",
            "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R",
            "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
            "x-ray": "X", "xray": "X", "yankee": "Y", "zulu": "Z",
            # Greek/phonetic variants
            "σιέρα": "S", "βίκτορ": "V", "μάικ": "M", "τάγκο": "T", "λίμα": "L", "νοβέμπερ": "N",
            "όσκαρ": "O", "πάπα": "P", "κίλο": "K", "τζουλιετ": "J", "γκόλφ": "G", "έκο": "E",
            "ντέλτα": "D", "γιάνκι": "Y", "ζουλου": "Z"
        }

        self.GREEK_NUMBER_MAP = {
            "μηδέν": "0", "ένα": "1", "δύο": "2", "τρία": "3", "τέσσερα": "4",
            "πέντε": "5", "έξι": "6", "επτά": "7", "οκτώ": "8", "εννέα": "9",
            "μία": "1", "δυο": "2", "τρεις": "3", "τεσσερα": "4", "εννιά": "9"
        }
        
        self.ENGLISH_NUMBER_MAP = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
        }
    
    def _create_default_config(self, config_file):
        """Create a default config.ini file"""
        default_config = configparser.ConfigParser()
        
        default_config['Audio'] = {
            'channels': '1',
            'input_channel': 'left',
            'rate': '44100',
            'chunk': '1024',
            'threshold': '500',
            'silence_time': '0.5',
            'audio_boost': '5.0',
            'output_volume': '1.0'
        }
        
        default_config['Paths'] = {
            'cancel_file': 'flags/cancel_ai.flag',
            'ai_ready_file': 'flags/ai_ready.flag',
            'loading_file': 'audio/system/loading_loop.wav',
            'ding_file': 'audio/system/ding.wav',
            'timeout_file': 'audio/system/timeout.wav',
            'ai_mode_file': 'audio/system/ai_mode.wav',
            'menu_file': 'audio/system/menu.wav',
            'repeater_info_file': 'audio/system/repeater_info.wav',
            'callsign_prompt_file': 'audio/system/callsign_prompt.wav'
        }
        
        default_config['CW'] = {
            'tone_freq': '800',
            'tone_duration': '0.2',
            'tone_volume': '1.0',
            'callsign': 'SV2TMT',
            'wpm': '20',
            'farnsworth_wpm': '',
            'id_interval': '600',
            'voice_id_interval': '1200'
        }
        
        default_config['Piper'] = {
            'model_path': 'models/el_GR-rapunzelina-low.onnx',
            'temp_audio': 'audio/temp/piper_temp.wav'
        }
        
        default_config['Weather'] = {
            'api_key': 'your_api_key_here',
            'city': 'Serres,GR',
            'cache_file': 'data/cache/weather_cache.json',
            'cache_duration': '900'
        }
        
        default_config['Satellite'] = {
            'tle_cache_file': 'data/cache/tle_cache.txt',
            'satellites': 'ISS (ZARYA), NOAA 15, NOAA 19, RADFXSAT (FOX-1B), SAUDISAT 1C (SO-50)',
            'tle_url1': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle',
            'tle_url2': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=tle',
            'tle_url3': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=amateur&FORMAT=tle'
        }
        
        default_config['Callsign'] = {
            'radioid_csv_url': 'https://radioid.net/static/user.csv',
            'radioid_local_file': 'data/cache/user.csv',
            'database_file': 'data/databases/callsigns.db'
        }
        
        default_config['Location'] = {
            'latitude': '41.08',
            'longitude': '23.55',
            'elevation': '50',
            'timezone': 'Europe/Athens'
        }
        
        default_config['Database'] = {
            'fun_facts_file': 'data/databases/fun_facts.txt'
        }
        
        default_config['Commands'] = {
            'max_commands': '4',
            'window_seconds': '60',
            'dtmf_prefix': ''
        }
        
        with open(config_file, 'w') as f:
            default_config.write(f)
        
        logger.info(f"Created default config file: {config_file}")

class HamRepeater:
    def __init__(self, args=None):
        self.config = RepeaterConfig(args=args)
        self.dt_detected = None
        self.morse_wpm = self.config.CW_WPM
        self.morse_farnsworth_wpm = self.config.CW_FARNSWORTH_WPM
        self.play_ai_mode = False
        self.ai_mode_running = False
        self.talking = False
        self.whisper_model = None
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
        self.command_times = deque()
        self.command_lock = threading.Lock()
        self.dtmf_buffer = []
        self.dtmf_buffer_timeout = 3.0
        self.last_dtmf_time = 0
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self._setup_audio_streams()
        
        # Initialize Piper voice
        self.piper_voice = None
        self._load_piper_voice()
        
        #Initialize Whisper
        self._initialize_whisper()
        
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
            
            # Output stream will be recreated dynamically based on channel selection
            self.output_stream = None
            
            logger.info("Audio streams initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio streams: {e}")
            raise
            
    def _initialize_whisper(self):
        """Initialize whisper model"""
        try:
            logger.info("Initializing Whisper model...")
            self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            logger.info("Whisper model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def play_audio(self, filename):
        """Play a WAV or MP3 file through the audio output"""
        try:
            # Clamp volume to valid range
            volume = max(0.0, min(5.0, self.config.OUTPUT_VOLUME))
            
            # Determine file type from extension
            file_ext = filename.lower().split('.')[-1]
            
            if file_ext == 'mp3':
                # MP3 handling
                from pydub import AudioSegment
                from pydub.playback import play
                
                audio = AudioSegment.from_mp3(filename)
                # Convert 0-5 scale to dB: 1.0->0dB, 2.0->+6dB, 5.0->+14dB
                if volume > 0:
                    volume_db = 20 * np.log10(volume)
                    audio = audio + volume_db
                else:
                    audio = audio - 60  # Effectively mute
                play(audio)
                
            elif file_ext == 'wav':
                # WAV handling
                with wave.open(filename, 'rb') as wf:
                    audio_stream = self.p.open(
                        format=self.p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )
                    
                    data = wf.readframes(self.config.CHUNK)
                    while data:
                        # Convert bytes to numpy array for volume adjustment
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        # Apply volume (with gain) and clip to prevent distortion
                        audio_data = np.clip(audio_data * volume, -32768, 32767).astype(np.int16)
                        # Write adjusted audio
                        audio_stream.write(audio_data.tobytes())
                        data = wf.readframes(self.config.CHUNK)
                    
                    audio_stream.close()
            else:
                logger.error(f"Unsupported audio format: {file_ext}")
                
        except FileNotFoundError:
            logger.error(f"Audio file {filename} not found!")
        except ImportError:
            logger.error("pydub library not installed. Install with: pip install pydub")
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
        """Generate and play a tone"""
        frequency = frequency or self.config.TONE_FREQ
        duration = duration or self.config.TONE_DURATION
        volume = volume or self.config.TONE_VOLUME
        
        try:
            # Calculate number of samples
            num_samples = int(self.config.RATE * duration)
            
            # Generate samples with smooth fade in/out
            samples = np.sin(2 * np.pi * np.arange(num_samples) * frequency / self.config.RATE)
            
            # Apply fade in/out to prevent audio clicks (5ms fade)
            fade_samples = int(0.005 * self.config.RATE)
            if num_samples > 2 * fade_samples:
                # Fade in
                samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out  
                samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            samples = (volume * samples).astype(np.float32)
            
            # Buffer size
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
        """Play morse code for a single character"""
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

    def play_text_morse(self, text, wpm=None, farnsworth_wpm=None):
        """Generate CW audio from text"""
        # Use provided WPM or fall back to instance settings or default
        current_wpm = wpm or getattr(self, 'morse_wpm', 20)
        current_farnsworth = farnsworth_wpm or getattr(self, 'morse_farnsworth_wpm', None)
        
        # Calculate timing based on WPM
        timing = self.calculate_timing(current_wpm, current_farnsworth)
        
        timing_info = f"WPM: {current_wpm}"
        if current_farnsworth:
            timing_info += f", Farnsworth: {current_farnsworth}"
        
        logger.info(f"Transmitting text {text} in CW ({timing_info})")
        
        frequency = self.config.TONE_FREQ
        volume = self.config.TONE_VOLUME
        rate = self.config.RATE
        
        audio_data = []
        
        for i, char in enumerate(text):
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
                if i < len(text) - 1:
                    next_char = text[i + 1] if i + 1 < len(text) else ''
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
                    
    def _load_piper_voice(self):
        """Load Piper voice model"""
        try:
            if os.path.exists(self.config.PIPER_MODEL_PATH):
                self.piper_voice = PiperVoice.load(self.config.PIPER_MODEL_PATH)
                logger.info(f"Piper voice loaded: {self.config.PIPER_MODEL_PATH}")
            else:
                logger.warning(f"Piper model not found: {self.config.PIPER_MODEL_PATH}")
                logger.warning("Download the Greek model with 'python -m piper.download_voices el_GR-rapunzelina-low' and put it in models/")
        except Exception as e:
            logger.error(f"Failed to load Piper voice: {e}")
            self.piper_voice = None
    
    def _split_sentences(self, text):
        """Split text into sentences for better TTS flow"""
        import re
        
        # Sentence endings
        sentence_endings = r'[.!?;,]\s+'
        
        # Split on sentence endings
        sentences = re.split(f'({sentence_endings})', text)
        
        # Recombine punctuation with sentences
        result = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i + 1]).strip()
            else:
                sentence = sentences[i].strip()
            
            if sentence:  # Only add non-empty sentences
                result.append(sentence)
        
        # Fallback: split very long texts at commas
        if len(result) <= 1 and len(text) > 150:
            parts = text.split(',')
            current = ""
            result = []
            for part in parts:
                if len(current + part) > 80 and current:
                    result.append(current.strip())
                    current = part
                else:
                    current += ("," if current else "") + part
            
            if current.strip():
                result.append(current.strip())
        
        return result if result else [text]

    def speak_with_piper(self, text):
        """TTS using Piper"""
        if not self.piper_voice:
            logger.error("Piper voice not loaded")
            return False
        
        try:
            # Split text into sentences
            sentences = self._split_sentences(text)
            logger.info(f"Speaking {len(sentences)} sentence(s)")
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                temp_file = f"audio/temp/piper_temp_{i}.wav"
                
                # Generate speech to WAV file
                with wave.open(temp_file, "wb") as wav_file:
                    self.piper_voice.synthesize_wav(sentence, wav_file)
                
                # Play audio file
                self.play_audio(temp_file)
                
                # Small pause between sentences (except for the last one)
                if i < len(sentences) - 1:
                    time.sleep(0.1)  # 100ms
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Error with Piper TTS: {e}")
            return False
    
    def run_ai_mode(self):
        """Run AI mode subprocess"""
        self.ai_mode_running = True
        timeout_played = [False]
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
        
        # Start subprocess
        try:
            import sys
            process = subprocess.Popen([sys.executable, "aimode.py"])
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
            
            # Process exited early
            loading_stop_event.set()
            loading_thread.join()
            self.play_audio(self.config.TIMEOUT_FILE)
            timeout_played[0] = True
            logger.warning("AI mode exited before ready signal")
        
        monitor_thread = threading.Thread(target=monitor_ai_ready, daemon=True)
        monitor_thread.start()
        
        # Wait for AI process to complete
        process.wait()
        monitor_thread.join(timeout=1.0)
        
        # Cleanup
        self.cleanup_files()
        
        # Only play timeout if exited normally
        # Don't play if it was an early exit
        if ai_ready[0] and not timeout_played[0]:
            while self.talking:
                time.sleep(0.5)
            self.play_audio(self.config.TIMEOUT_FILE)
        
        self.ai_mode_running = False
        logger.info("AI mode completed")
        
    def download_file(self, url, destination, chunk_size=8192):
        """Download a file with a live progress bar."""
        import requests
        import sys

        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        bar_width = 40

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        percent = downloaded / total
                        filled = int(bar_width * percent)
                        bar = "█" * filled + "░" * (bar_width - filled)
                        sys.stdout.write(f"\rDownloading File: [{bar}] {int(percent*100)}%")
                        sys.stdout.flush()

        sys.stdout.write("\rDownloading File: [" + "█"*bar_width + "] 100%\n")
        sys.stdout.flush()
        print("Download complete!")
    
    def load_radioid_data(self, force_update=False):
        """Load RadioID database"""
        import csv
        import requests
        import sqlite3

        db_file = self.config.CALLSIGN_DB_FILE
        csv_file = self.config.RADIOID_LOCAL_FILE
        url = self.config.RADIOID_CSV_URL

        # Determine whether a fresh CSV download is needed
        download_needed = True
        if os.path.exists(csv_file) and not force_update:
            file_age = time.time() - os.path.getmtime(csv_file)
            if file_age < 86400:  # 24 hours
                logger.info("Using cached RadioID user.csv (less than 1 day old)")
                download_needed = False
            else:
                logger.info("Cached RadioID user.csv is older than 1 day - redownloading...")
        elif force_update:
            logger.info("Force update enabled — downloading fresh RadioID user.csv")

        try:
            # Download CSV if needed
            if download_needed or force_update:
                logger.info("Downloading fresh user.csv from RadioID (This may take a while...)")
                self.download_file(url, csv_file)
                logger.info("Downloaded fresh user.csv from RadioID")

            # Rebuild SQLite database if CSV is newer than DB or DB doesn't exist
            rebuild_needed = (
                not os.path.exists(db_file)
                or force_update
                or os.path.getmtime(csv_file) > os.path.getmtime(db_file)
            )

            if rebuild_needed:
                logger.info("Updating SQLite callsign database from CSV...")
                conn = sqlite3.connect(db_file)
                conn.execute("PRAGMA journal_mode = OFF;")
                conn.execute("PRAGMA synchronous = OFF;")

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS callsigns (
                        callsign TEXT PRIMARY KEY,
                        first_name TEXT,
                        last_name TEXT,
                        city TEXT,
                        country TEXT
                    )
                ''')

                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    data = [
                        (
                            row["CALLSIGN"].strip().upper(),
                            row["FIRST_NAME"].strip(),
                            row["LAST_NAME"].strip(),
                            row["CITY"].strip(),
                            row["COUNTRY"].strip()
                        )
                        for row in reader
                    ]

                conn.executemany('INSERT OR REPLACE INTO callsigns VALUES (?, ?, ?, ?, ?)', data)
                conn.commit()
                conn.close()
                logger.info("SQLite callsign database updated.")
            else:
                logger.info("SQLite database is already up-to-date with the cached CSV.")

        except Exception as e:
            logger.warning(f"Could not download/process RadioID CSV: {e}")

        return db_file
        
    def lookup_callsign_in_db(self, callsign):
        """Look up callsign in RadioID database"""
        import sqlite3
        
        try:
            conn = sqlite3.connect(self.callsign_data)
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
    
    def command_allowed(self):
        """Command watchdog"""
        now = time.time()

        with self.command_lock:
            # Remove timestamps outside the time window
            while self.command_times and now - self.command_times[0] > self.config.COMMAND_WINDOW:
                self.command_times.popleft()

            if len(self.command_times) >= self.config.MAX_COMMANDS:
                logger.warning(f"Command rate limit exceeded {self.config.MAX_COMMANDS}/{self.config.COMMAND_WINDOW}s")
                while self.talking:
                    time.sleep(0.5)
                self.speak_with_piper(f"Παρακαλώ περιμένετε {int(round((self.command_times[0] + self.config.COMMAND_WINDOW) - now))} δευτερόλεπτα.")
                return False

            self.command_times.append(now)
            return True
    
    def command_schedule(self, attr, message):
        if not self.ai_mode_running: # Run only outside AI mode
            if not getattr(self, attr):
                setattr(self, attr, True)
                logger.info(message)
    
    def dtmf_listener(self):
        """Listen for DTMF tones and handle commands"""
        
        def ai_command():
            if self.ai_mode_running:
                try:
                    with open(self.config.CANCEL_FILE, "w") as f:
                        f.write("cancel")
                    logger.info("AI mode cancel signal sent")
                except Exception as e:
                    logger.error(f"Failed to write cancel file: {e}")
            else:
                self.play_ai_mode = True
                logger.info("AI mode activation requested")

        command_map = {
            '*': ai_command,
            '#': lambda: self.command_schedule("play_menu",    "Menu playback scheduled"),
            '0': lambda: self.command_schedule("play_info",    "Repeater info playback scheduled"),          
            '1': lambda: self.command_schedule("play_time",    "Time and date playback scheduled"),
            '2': lambda: self.command_schedule("play_weather", "Weather playback scheduled"),
            '3': lambda: self.command_schedule("play_band",    "Band conditions playback scheduled"),
            '4': lambda: self.command_schedule("play_fact",    "Random fun fact playback scheduled"),            
            '5': lambda: self.command_schedule("lookup_callsign", "Callsign lookup scheduled"),
            '6': lambda: self.command_schedule("play_satpass", "Satellite pass playback scheduled"),
            '7': lambda: self.command_schedule("play_meme",    "Random meme playback scheduled"),
        }

        while True:
            try:
                detected = detect_dtmf()
                if detected:
                    self.dt_detected = detected
                    current_time = time.time()
                    
                    # Reset buffer if timeout exceeded
                    if current_time - self.last_dtmf_time > self.dtmf_buffer_timeout:
                        self.dtmf_buffer = []
                    
                    self.last_dtmf_time = current_time
                    self.dtmf_buffer.append(detected)
                    
                    # Build current sequence
                    current_sequence = ''.join(self.dtmf_buffer)
                    if self.config.DTMF_PREFIX:
                        logger.info(f"DTMF detected: {detected} (sequence: {current_sequence})")
                    else:
                        logger.info(f"DTMF detected: {detected}")
                    
                    # Initialize command to execute
                    command_to_execute = None
                    
                    # If prefix is required
                    if self.config.DTMF_PREFIX:
                        # Check if we have the complete prefix with the command
                        if len(current_sequence) >= len(self.config.DTMF_PREFIX) + 1:
                            # Extract prefix and command
                            received_prefix = current_sequence[:len(self.config.DTMF_PREFIX)]
                            command = current_sequence[len(self.config.DTMF_PREFIX)]
                            
                            if received_prefix == self.config.DTMF_PREFIX:
                                # Valid prefix, set command to execute
                                command_to_execute = command
                            else:
                                logger.info(f"Invalid prefix: {received_prefix} (expected: {self.config.DTMF_PREFIX})")
                            
                            # Reset buffer
                            self.dtmf_buffer = []
                        
                        # Check if current sequence is too long (invalid)
                        elif len(current_sequence) > len(self.config.DTMF_PREFIX) * 2:
                            logger.warning(f"DTMF sequence too long without valid prefix, resetting")
                            self.dtmf_buffer = []
                    
                    else:
                        # Normal command mode
                        command_to_execute = detected
                        self.dtmf_buffer = []
                    
                    # Execute command if one was determined
                    if command_to_execute:
                        # Check if this command is disabled
                        if command_to_execute in self.config.DISABLED_DTMF_COMMANDS:
                            logger.info(f"DTMF command {command_to_execute} is disabled, ignoring")
                            continue

                        handler = command_map.get(command_to_execute)
                        if handler:
                            if self.command_allowed():
                                handler()

            except Exception as e:
                logger.error(f"Error in DTMF listener: {e}")
                time.sleep(1)
    
    def callsign_thread(self):
        """Periodic callsign CW ID"""
        while True:
            time.sleep(self.config.CW_ID_INTERVAL)
            # Wait if currently talking
            while self.talking:
                time.sleep(1)
            self.play_text_morse(self.config.CALLSIGN)
            
    def voiceid_thread(self):
        """Periodic voice/audio ID"""
        while True:
            time.sleep(self.config.VOICE_ID_INTERVAL)
            # Wait if currently talking
            while self.talking:
                time.sleep(1)
            self.play_audio(self.config.REPEATER_INFO_FILE)
            
    def fetch_tles(self):
        """Fetch TLE data for satellite passses"""
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
        """Record audio clip for callsign search"""
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
        """Speech To Text conversion with FasterWhisper"""
        try:
            segments, _ = self.whisper_model.transcribe(
                filename, 
                language="en"
            )
            result = "".join([seg.text for seg in segments]).strip().upper()
            logger.info(f"Transcribed callsign input: {result}")
            return result
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""
            
    def extract_callsign_from_text(self, text):
        """Callsign identification from text"""
        words = re.sub(r'[^\w\s]', ' ', text).lower().split()
        callsign = ""
        
        for word in words:
            # Direct phonetic alphabet mapping
            if word in self.config.PHONETIC_MAP:
                callsign += self.config.PHONETIC_MAP[word]
            elif word in self.config.GREEK_NUMBER_MAP:
                callsign += self.config.GREEK_NUMBER_MAP[word]
            elif word in self.config.ENGLISH_NUMBER_MAP:
                callsign += self.config.ENGLISH_NUMBER_MAP[word]
            elif word[0].isdigit():
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
                    # Fallback: take first letter if it looks like speech
                    if word.isalpha() and len(word) > 1:
                        callsign += word[0].upper()
        
        # Clean up common patterns and validate
        callsign = callsign.replace(" ", "").upper()
        
        # Basic callsign validation, should have 4 alphanumeric characters
        if len(callsign) >= 4 and any(c.isalpha() for c in callsign):
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
        dtmf_thread = None
        if self.config.ENABLE_DTMF:
            dtmf_thread = threading.Thread(target=self.dtmf_listener, daemon=True)
            dtmf_thread.start()

        callsign_thread = None
        if self.config.ENABLE_CW_ID:
            callsign_thread = threading.Thread(target=self.callsign_thread, daemon=True)
            callsign_thread.start()
            
        voiceid_thread = None
        if self.config.ENABLE_VOICE_ID:
            voiceid_thread = threading.Thread(target=self.voiceid_thread, daemon=True)
            voiceid_thread.start()

        logger.info("Ham repeater running...")
        
        try:
            while True:
                # Read audio input
                data = self.input_stream.read(self.config.CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # Handle channel selection for stereo input
                output_channels = self.config.CHANNELS  # Default to input channels
                
                if self.config.CHANNELS == 2:
                    # Reshape to separate channels
                    audio_np = audio_np.reshape(-1, 2)
                    
                    if self.config.INPUT_CHANNEL == 'left':
                        audio_np = audio_np[:, 0]  # Keep only left channel
                        output_channels = 1  # Output as mono
                    elif self.config.INPUT_CHANNEL == 'right':
                        audio_np = audio_np[:, 1]  # Keep only right channel
                        output_channels = 1  # Output as mono
                    elif self.config.INPUT_CHANNEL == 'mono':
                        audio_np = np.mean(audio_np, axis=1).astype(np.int16)  # Average both
                        output_channels = 1  # Output as mono
                    elif self.config.INPUT_CHANNEL == 'both':
                        # Keep stereo - flatten back
                        audio_np = audio_np.flatten()
                        output_channels = 2  # Output as stereo
                    else:
                        logger.warning(f"Unknown input_channel: {self.config.INPUT_CHANNEL}, using both")
                        audio_np = audio_np.flatten()
                        output_channels = 2
                
                # Apply audio boost and clipping
                audio_np = np.clip(
                    audio_np * self.config.AUDIO_BOOST, 
                    -32768, 32767
                ).astype(np.int16)
                boosted_data = audio_np.tobytes()
                
                # Check if audio is above threshold (someone talking)
                if np.max(np.abs(audio_np)) > self.config.THRESHOLD:
                    if self.config.ENABLE_AUDIO_REPEAT:
                        # Recreate output stream if needed with correct channels
                        if self.output_stream is None or self.output_stream._channels != output_channels:
                            if self.output_stream:
                                self.output_stream.close()
                            self.output_stream = self.p.open(
                                format=self.config.FORMAT,
                                channels=output_channels,
                                rate=self.config.RATE,
                                output=True
                            )
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
                            """Feature/Module logic"""
                            # Only run modules if there is silence
                            # Only play roger beep if enabled and AI mode is not about to start
                            if self.config.ENABLE_ROGER_BEEP and not (self.play_ai_mode and not self.ai_mode_running):
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
                                    meme_files = [f for f in os.listdir(meme_folder) if f.lower().endswith(".wav") or f.lower().endswith(".mp3")]
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
                            
                            # Fun Fact playback
                            if self.play_fact:
                                try:
                                    fact_file = self.config.FUN_FACTS_FILE
                                    if os.path.exists(fact_file):
                                        with open(fact_file, "r", encoding="utf-8") as f:
                                            facts = [line.strip() for line in f if line.strip()]
                                        if facts:
                                            fact = random.choice(facts)
                                            logger.info(f"Speaking fun fact: {fact}")
                                            self.speak_with_piper(f"Fun fact: {fact}")
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

                                    hour_word = self.config.GREEK_HOUR_NAMES.get(hour, str(hour))
                                    minute_word = f"{minute}" if minute != 0 else "ακριβώς"

                                    if minute != 0:
                                        time_phrase = f"{hour_word} και {minute_word}"
                                    else:
                                        time_phrase = f"{hour_word} ακριβώς"

                                    date_phrase = f"Σήμερα είναι {day} {month_gr} του {year}."
                                    full_phrase = f"Η ώρα είναι {time_phrase}. {date_phrase}"

                                    logger.info(f"Speaking time and date: {full_phrase}")

                                    self.speak_with_piper(full_phrase)

                                except Exception as e:
                                    logger.error(f"Error during time and date playback: {e}")
                                finally:
                                    self.play_time = False
                                    
                            # Weather playback
                            if self.play_weather:
                                try:
                                    import requests, json
                                    from datetime import datetime

                                    city = self.config.WEATHER_CITY
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
                                            windspeed = data["wind"]["speed"]
                                            city = data["name"]
                                            
                                            # Convert m/s to beaufort
                                            BEAUFORT_LIMITS = [0.5, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
                                            
                                            beaufort = 0
                                            for limit in BEAUFORT_LIMITS:
                                                if windspeed <= limit:
                                                    break
                                                beaufort += 1  

                                            # Construct forecast message
                                            forecast = (
                                                f"Ο καιρός στην περιοχή {city} είναι {description}, "
                                                f"με θερμοκρασία {temp} βαθμούς Κελσίου, "
                                                f"υγρασία {humidity} τοις εκατό, "
                                                f"και ανέμους {beaufort} μποφόρ."
                                            )

                                            # Save to cache
                                            with open(self.config.WEATHER_CACHE_FILE, "w", encoding="utf-8") as f:
                                                json.dump({"forecast": forecast, "timestamp": time.time()}, f)
                                        else:
                                            logger.warning(f"Failed to fetch weather: {response.status_code}")
                                            forecast = "Δεν μπόρεσα να ανακτήσω τα δεδομένα καιρού."

                                    logger.info(f"Speaking weather: {forecast}")
                                    self.speak_with_piper(forecast)

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
                                        
                                        greek_condition_names = {
                                            "Good": "καλές",
                                            "Fair": "μέτριες",
                                            "Poor": "κακές",
                                        }

                                        for key, condition in band_reports.items():
                                            greek_band = greek_band_names.get(key, key)
                                            greek_condition = greek_condition_names.get(condition, condition)
                                            band_phrases.append(f"{greek_band}: {greek_condition}")

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
                                        self.speak_with_piper(full_report)
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
                                    from skyfield.api import EarthSatellite, Topos, load, Loader, utc

                                    # Observer position
                                    observer = Topos(latitude_degrees=self.config.LATITUDE, 
                                                     longitude_degrees=self.config.LONGITUDE, 
                                                     elevation_m=self.config.ELEVATION)
                                    local_tz = pytz.timezone(self.config.TIMEZONE)
                                    ts = load.timescale()
                                    now = ts.now()
                                    t_end = ts.utc((datetime.utcnow() + timedelta(hours=24)).replace(tzinfo=utc))

                                    # Fetch TLEs
                                    tle_text = self.fetch_tles()
                                    lines = tle_text.strip().splitlines()

                                    # Parse TLEs into Skyfield satellites
                                    satellites = []
                                    
                                    # Name matching
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

                                    load = Loader("data/cache") # Initialize custom path
                                    eph = load("de421.bsp")  # Required by skyfield
                                    
                                    # Find upcoming AOS events for each satellite
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

                                        for name, dt in passes[:3]:
                                            hour = dt.hour % 12
                                            minute = dt.minute
                                            hour_str = self.config.GREEK_HOUR_NAMES.get(hour, str(hour))
                                            minute_str = f"{minute:02d}"
                                            spoken_parts.append(f"{name}: {hour_str} και {minute_str}")

                                        spoken = "Οι επόμενες διελεύσεις είναι: " + ", ".join(spoken_parts) + " ώρα Ελλάδος."

                                    logger.info(f"Speaking satellite passes: {spoken}")
                                    self.speak_with_piper(spoken)

                                except Exception as e:
                                    logger.error(f"Error during satellite pass playback: {e}")
                                finally:
                                    self.play_satpass = False
                                    
                            # Callsign lookup
                            if self.lookup_callsign:
                                try:
                                    self.play_audio(self.config.CALLSIGN_PROMPT_FILE)
                                    audio_file = self.record_callsign_audio(duration=8)
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
                                    self.speak_with_piper(message)
                                    
                                    # Clean up audio file
                                    if os.path.exists(audio_file):
                                        os.remove(audio_file)

                                except Exception as e:
                                    logger.error(f"Error during callsign lookup: {e}")
                                    # Play error message
                                    error_msg = "Σφάλμα κατά την αναζήτηση διακριτικού."
                                    self.speak_with_piper(error)
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
    parser = argparse.ArgumentParser(
        description='ChatRF: AI-Enhanced Ham Radio Repeater',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DTMF Command List:
  *  - AI Mode (Enable/disable)
  #  - Menu
  0  - Repeater Info
  1  - Time and Date
  2  - Weather
  3  - Band Conditions
  4  - Random Fun Fact
  5  - Callsign Lookup
  6  - Satellite Passes
  7  - Random Meme

Examples:
  # Disable AI mode and weather commands
  python repeater.py --disable-dtmf * 2
  
  # Disable all number commands but keep * and #
  python repeater.py --disable-dtmf 0 1 2 3 4 5 6 7
        """
    )
    
    parser.add_argument(
        '--no-audio-repeat',
        action='store_true',
        help='Disable audio repeating (monitor mode only)'
    )
    
    parser.add_argument(
        '--no-roger',
        action='store_true',
        help='Disable roger beep at end of transmission'
    )
    
    parser.add_argument(
        '--no-cw-id',
        action='store_true',
        help='Disable periodic CW callsign identification'
    )
    
    parser.add_argument(
        '--no-voice-id',
        action='store_true',
        help='Disable periodic voice/audio station identification'
    )
    
    parser.add_argument(
        '--disable-dtmf',
        nargs='+',
        metavar='CMD',
        help='Disable specific DTMF commands (e.g., --disable-dtmf * 2 5 to disable AI mode, weather, and callsign lookup)'
    )
    
    parser.add_argument(
        '--no-dtmf',
        action='store_true',
        help='Disable DTMF command listening'
    )
    
    parser.add_argument(
        '--no-ptt',
        action='store_true',
        help='Disable PTT control via Raspberry Pi GPIO (REQUIRED on Windows)'
    )
    
    parser.add_argument(
        '--audio-boost',
        type=float,
        help='Audio boost factor'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        help='Audio detection threshold'
    )
    
    args = parser.parse_args()
    
    ensure_directories()
    repeater = HamRepeater(args)
    
    # Apply additional args
    if args.audio_boost:
        repeater.config.AUDIO_BOOST = args.audio_boost
    if args.threshold:
        repeater.config.THRESHOLD = args.threshold
    
    banner = r"""   
                                                                                                    
          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                                                         
        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                                                       
       ▓▓▓▓▓▓  ▓▓▓▓▓▓▓  ▓▓▓▓▓▓                                                                      
       ▓▓▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓▓▓      ███████   ██                     █    ████████   █████████      
       ▓▓▓▓▓ ▓▓ ▓   ▓ ▓▓ ▓▓▓▓▓    ███    ███  ██                    ██    ██     ██  ███            
       ▓▓▓▓▓ ▓▓▓ ▓ ▓ ▓▓▓ ▓▓▓▓▓    ██          ████████   ███████  ██████  ██     ███ ███            
       ▓▓▓▓▓▓  ▓▓▓ ▓▓▓  ▓▓▓▓▓▓    ██          ███   ███       ███   ██    █████████  █████████      
       ▓▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓    ██      ██  ██    ███  ████████   ██    ██   ███   ███            
       ▓▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓    ████   ███  ██    ███ ██    ███   ██    ██    ███  ███            
       ▓▓▓▓▓▓▓▓▓     ▓▓▓▓▓▓▓▓▓      ██████    ██    ███  ████████   ████  ██     ███ ███            
         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                                                        
            ▓▓▓▓▓                                                                                   
            ▓▓▓                                                                                     
                                                                                                    
                                       ChatRF: AI-Enhanced Repeater
                                                 By SV2TMT
    ----------------------------------------------------------------------------------------------
    """
    print(banner)
    
    # Log configuration
    logger.info("Starting repeater with configuration:")
    logger.info(f"  Audio Repeat: {repeater.config.ENABLE_AUDIO_REPEAT}")
    logger.info(f"  Roger Beep: {repeater.config.ENABLE_ROGER_BEEP}")
    logger.info(f"  CW ID: {repeater.config.ENABLE_CW_ID}")
    logger.info(f"  Voice ID: {repeater.config.ENABLE_VOICE_ID}")
    logger.info(f"  DTMF Commands: {repeater.config.ENABLE_DTMF}")
    logger.info(f"  PTT Control: {repeater.config.ENABLE_PTT}")
    logger.info(f"  Audio Boost: {repeater.config.AUDIO_BOOST}")
    logger.info(f"  Threshold: {repeater.config.THRESHOLD}")
    if repeater.config.DISABLED_DTMF_COMMANDS:
        logger.info(f"  Disabled DTMF Commands: {', '.join(sorted(repeater.config.DISABLED_DTMF_COMMANDS))}")
    else:
        logger.info(f"  DTMF Commands: All enabled")
    
    # PTT via Raspberry Pi GPIO
    if repeater.config.ENABLE_PTT:
        from ptt_audio_monitor import ptt_for_repeater
        repeater = ptt_for_repeater(repeater, ptt_pin=18)
    
    try:
        repeater.run_repeater()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        repeater.shutdown()

if __name__ == "__main__":
    main()
