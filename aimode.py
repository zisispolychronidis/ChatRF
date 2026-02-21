import pyaudio
import wave
import numpy as np
import ollama
import os
import time
import subprocess
import logging
import threading
import json
import configparser
import re
import platform
import threading
import requests
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from contextlib import contextmanager
from piper import PiperVoice
# Detect platform
IS_UNIX = platform.system() != 'Windows'

if IS_UNIX:
    import signal

# ANSI colors
LOG_COLORS = {
    'DEBUG': '\033[90m',     # Bright Black / Gray
    'INFO': '\033[36m',      # Green
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

class TimeoutError(Exception):
    """Custom exception for timeouts"""
    pass

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
        'logs/aimode.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Error file log
    error_handler = RotatingFileHandler(
        'logs/aimode_errors.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger

logger = setup_logging()

def call_with_timeout(func, args=(), kwargs=None, timeout=45):
    """
    Call a function with a timeout (cross-platform).
    Uses signal on Unix/Linux, threading on Windows.
    
    Args:
        func: Function to call
        args: Tuple of positional arguments
        kwargs: Dict of keyword arguments
        timeout: Timeout in seconds
        
    Returns:
        Function result or raises TimeoutError
    """
    if kwargs is None:
        kwargs = {}
    
    if IS_UNIX:
        # Unix: Signal-based timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    else:
        # Windows: Threading-based timeout
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]

class AIConfig:
    """Configuration class for AI mode settings"""
    def __init__(self, config_file='config/settings/aimode_config.ini'):
        self.config = configparser.ConfigParser()
        
        # Check if config file exists
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found! Creating default config...")
            self._create_default_config(config_file)
        
        self.config.read(config_file)
        
        # Audio Settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = self.config.getint('Audio', 'channels', fallback=1)
        self.INPUT_CHANNEL = self.config.get('Audio', 'input_channel', fallback='left').lower()
        self.RATE = self.config.getint('Audio', 'rate', fallback=44100)
        self.CHUNK = self.config.getint('Audio', 'chunk', fallback=1024)
        self.THRESHOLD = self.config.getint('Audio', 'threshold', fallback=500)
        self.MIN_TALKING = self.config.getfloat('Audio', 'min_talking', fallback=0.2)
        self.OUTPUT_VOLUME = self.config.getfloat('Audio', 'output_volume', fallback=1.0)
        self.INPUT_DEVICE = self.config.getint('Audio', 'input_device', fallback=-1)
        self.OUTPUT_DEVICE = self.config.getint('Audio', 'output_device', fallback=-1)
        
        # File paths
        self.OUTPUT_FILE = self.config.get('Paths', 'output_file', fallback='audio/temp/recorded_audio.wav')
        self.CANCEL_FILE = self.config.get('Paths', 'cancel_file', fallback='flags/cancel_ai.flag')
        self.READY_FILE = self.config.get('Paths', 'ready_file', fallback='flags/ai_ready.flag')
        self.CONTEXT_FILE = self.config.get('Paths', 'context_file', fallback='data/context/conversation_context.json')
        self.SYSTEM_PROMPT_FILE = self.config.get('Paths', 'system_prompt_file', fallback='config/prompts/system_prompt.txt')
        
        # AI Model Settings
        self.WHISPER_MODEL_SIZE = self.config.get('AI', 'whisper_model_size', fallback='small')
        self.OLLAMA_MODEL_NAME = self.config.get('AI', 'ollama_model', fallback='gemma3')
        self.TEMPERATURE = self.config.getfloat('AI', 'temperature', fallback=0.2)
        self.AUTO_ADJUST_TIMEOUT = self.config.getboolean('AI', 'auto_adjust_timeout', fallback=True)
        self.PRELOAD_MODEL = self.config.getboolean('AI', 'preload_model', fallback=False)

        # Ollama Server Settings
        self.OLLAMA_HOST = self.config.get('AI', 'ollama_host', fallback='localhost')
        self.OLLAMA_PORT = self.config.getint('AI', 'ollama_port', fallback=11434)
        self.OLLAMA_USE_HTTPS = self.config.getboolean('AI', 'ollama_use_https', fallback=False)
        self.OLLAMA_API_KEY = self.config.get('AI', 'ollama_api_key', fallback='')  # For future auth support
        
        # Timing Settings
        self.DEFAULT_TIMEOUT = self.config.getint('Timing', 'default_timeout', fallback=60)
        self.SILENCE_LIMIT_SECONDS = self.config.getint('Timing', 'silence_limit_seconds', fallback=2)
        self.LLM_TIMEOUT = self.config.getint('Timing', 'llm_timeout', fallback=45)
        self.LLM_CONNECT_TIMEOUT = self.config.getint('Timing', 'llm_connect_timeout', fallback=10)
        
        # Context Settings
        self.MAX_CONTEXT_MESSAGES = self.config.getint('Context', 'max_context_messages', fallback=20)
        self.CONTEXT_TIMEOUT_MINUTES = self.config.getint('Context', 'context_timeout_minutes', fallback=30)
        
        # Piper TTS Settings
        self.PIPER_MODEL_PATH = self.config.get('Piper', 'model_path', fallback='models/el_GR-rapunzelina-low.onnx')
        self.PIPER_TEMP_AUDIO = self.config.get('Piper', 'temp_audio', fallback='audio/temp/piper_ai_temp.wav')
        
        # Thinking Sound Settings
        self.THINKING_SOUND_ENABLED = self.config.getboolean('ThinkingSound', 'enabled', fallback=True)
        self.THINKING_SOUND_VOLUME = self.config.getfloat('ThinkingSound', 'volume', fallback=0.6)
    
    def _create_default_config(self, config_file):
        """Create a default aimode_config.ini file"""
        default_config = configparser.ConfigParser()
        
        default_config['Audio'] = {
            'channels': '1',
            'input_channel': 'left',
            'rate': '44100',
            'chunk': '1024',
            'threshold': '500',
            'min_talking': '0.2',
            'output_volume': '1.0',
            'input_device': '-1',
            'output_device': '-1'
        }
        
        default_config['Paths'] = {
            'output_file': 'audio/temp/recorded_audio.wav',
            'cancel_file': 'flags/cancel_ai.flag',
            'ready_file': 'flags/ai_ready.flag',
            'context_file': 'data/context/conversation_context.json',
            'system_prompt_file': 'config/prompts/system_prompt.txt'
        }
        
        default_config['AI'] = {
            'whisper_model_size': 'small',
            'ollama_model': 'gemma3',
            'temperature': '0.2',
            'ollama_host': 'localhost',
            'ollama_port': '11434',
            'ollama_use_https': 'false',
            'ollama_api_key': '',
            'auto_adjust_timeout': 'true',
            'preload_model': 'false'
        }
        
        default_config['Timing'] = {
            'default_timeout': '60',
            'silence_limit_seconds': '2',
            'llm_timeout': '45',
            'llm_connect_timeout': '10'
        }
        
        default_config['Context'] = {
            'max_context_messages': '20',
            'context_timeout_minutes': '30'
        }
        
        default_config['Piper'] = {
            'model_path': 'models/el_GR-rapunzelina-low.onnx',
            'temp_audio': 'audio/temp/piper_ai_temp.wav'
        }
        
        default_config['ThinkingSound'] = {
            'enabled': 'true',
            'volume': '0.6'
        }
        
        with open(config_file, 'w') as f:
            default_config.write(f)
        
        logger.info(f"Created default config file: {config_file}")
    
    def get_ollama_base_url(self):
        """Build the Ollama API base URL from configuration"""
        protocol = 'https' if self.OLLAMA_USE_HTTPS else 'http'
        return f"{protocol}://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

    def load_system_prompt(self):
        """Load system prompt from file"""
        try:
            with open(self.SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"System prompt file {self.SYSTEM_PROMPT_FILE} not found, using default")
            return "Είσαι ένας AI βοηθός ενσωματωμένος σε έναν ραδιοερασιτεχνικό σταθμό (ham radio) που βρίσκεται στις Σέρρες, Ελλάδα. Ο κύριος ρόλος σου είναι να βοηθάς τους ραδιοερασιτέχνες απαντώντας σε ερωτήσεις."
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return "Είσαι ένας AI βοηθός ενσωματωμένος σε έναν ραδιοερασιτεχνικό σταθμό (ham radio) που βρίσκεται στις Σέρρες, Ελλάδα. Ο κύριος ρόλος σου είναι να βοηθάς τους ραδιοερασιτέχνες απαντώντας σε ερωτήσεις."

class OllamaClient:
    """
    Enhanced client for communicating with Ollama servers
    - Handles connection, timeout, retries, and error handling
    - Detects if model needs to be loaded and adjusts timeout
    - Caches model loading status for performance
    """
    
    def __init__(self, config):
        self.config = config
        self.base_url = config.get_ollama_base_url()
        self.session = requests.Session()
        
        # Cache for loaded model status
        self.loaded_models_cache = set()
        self.last_model_check = {}
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ChatRF/1.0'
        })
        
        # Add API key if configured
        if config.OLLAMA_API_KEY:
            self.session.headers.update({
                'Authorization': f'Bearer {config.OLLAMA_API_KEY}'
            })
    
    def test_connection(self):
        """Test if Ollama server is accessible"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self.config.LLM_CONNECT_TIMEOUT
            )
            response.raise_for_status()
            logger.info(f"✓ Connected to Ollama server at {self.base_url}")
            return True
        except requests.ConnectionError:
            logger.error(f"✗ Cannot connect to Ollama server at {self.base_url}")
            return False
        except requests.Timeout:
            logger.error(f"✗ Connection timeout to Ollama server at {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"✗ Error testing Ollama connection: {e}")
            return False
    
    def normalize_model_name(self, model_name):
        # If model name doesn't have a tag, add :latest
        if ':' not in model_name:
            return f"{model_name}:latest"
        return model_name

    def list_models(self):
        """List available models on the server"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self.config.LLM_CONNECT_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_running_models(self):
        """
        Get list of currently loaded/running models
        
        Returns:
            dict: Dictionary mapping model names to their info
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/ps",
                timeout=self.config.LLM_CONNECT_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse running models
            running = {}
            for model_info in data.get('models', []):
                model_name = model_info.get('name', '')
                running[model_name] = {
                    'size': model_info.get('size', 0),
                    'size_vram': model_info.get('size_vram', 0),
                    'expires_at': model_info.get('expires_at', ''),
                }
            
            return running
            
        except Exception as e:
            logger.warning(f"Could not get running models: {e}")
            return {}
    
    def is_model_loaded(self, model_name):
        """
        Check if a specific model is currently loaded in memory
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            bool: True if model is loaded, False otherwise
        """
        try:
            # Normalize the model name first
            normalized_name = self.normalize_model_name(model_name)
            
            # Check cache first (valid for 30 seconds)
            cache_key = normalized_name
            if cache_key in self.last_model_check:
                last_check_time = self.last_model_check[cache_key]
                if time.time() - last_check_time < 30:
                    return normalized_name in self.loaded_models_cache
            
            # Query server for running models
            running_models = self.get_running_models()
            
            # Update cache
            self.loaded_models_cache = set(running_models.keys())
            self.last_model_check[cache_key] = time.time()
            
            is_loaded = normalized_name in running_models
            
            if is_loaded:
                logger.info(f"✓ Model '{normalized_name}' is already loaded in memory")
            else:
                logger.info(f"⚠ Model '{normalized_name}' needs to be loaded (will take extra time)")
            
            return is_loaded
            
        except Exception as e:
            logger.warning(f"Could not check if model is loaded: {e}")
            # Assume not loaded to be safe (use longer timeout)
            return False
    
    def estimate_model_load_time(self, model_name):
        """
        Estimate how long it will take to load a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            int: Estimated load time in seconds
        """
        # Get model info to estimate size
        try:
            # Try to get model details
            models = self.list_models()
            
            # Simple heuristic based on model name
            model_lower = model_name.lower()
            
            if any(size in model_lower for size in ['120b', '70b', '65b', '72b']):
                return 120  # Large models: 2 minutes
            elif any(size in model_lower for size in ['30b', '34b']):
                return 90   # Medium-large: 1.5 minutes
            elif any(size in model_lower for size in ['13b', '14b']):
                return 60   # Medium: 1 minute
            elif any(size in model_lower for size in ['7b', '8b']):
                return 45   # Small-medium: 45 seconds
            elif any(size in model_lower for size in ['3b', '2b', '1b']):
                return 30   # Small: 30 seconds
            else:
                return 60   # Default: 1 minute
                
        except Exception as e:
            logger.warning(f"Could not estimate load time: {e}")
            return 60  # Default to 1 minute
    
    def get_dynamic_timeout(self, model_name):
        """
        Calculate appropriate timeout based on whether model is loaded
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            int: Timeout in seconds
        """
        base_timeout = self.config.LLM_TIMEOUT
        
        if self.is_model_loaded(model_name):
            # Model is loaded, use normal timeout
            logger.debug(f"Using normal timeout: {base_timeout}s")
            return base_timeout
        else:
            # Model needs loading, add load time to timeout
            load_time = self.estimate_model_load_time(model_name)
            total_timeout = base_timeout + load_time
            logger.info(f"Model needs loading - using extended timeout: {total_timeout}s (base: {base_timeout}s + load: {load_time}s)")
            return total_timeout
    
    def chat(self, messages, stream=False, auto_adjust_timeout=True):
        """
        Send a chat request to Ollama
        
        Args:
            messages: List of message dicts
            stream: Whether to stream the response
            auto_adjust_timeout: Automatically adjust timeout for model loading
            
        Returns:
            Response dict or generator if streaming
            
        Raises:
            requests.Timeout: If request times out
            requests.RequestException: For other request errors
        """
        try:
            payload = {
                'model': self.config.OLLAMA_MODEL_NAME,
                'messages': messages,
                'options': {
                    'temperature': self.config.TEMPERATURE
                },
                'stream': stream
            }
            
            # Determine timeout
            if auto_adjust_timeout:
                read_timeout = self.get_dynamic_timeout(self.config.OLLAMA_MODEL_NAME)
            else:
                read_timeout = self.config.LLM_TIMEOUT
            
            # Use tuple for timeout: (connect_timeout, read_timeout)
            timeout = (self.config.LLM_CONNECT_TIMEOUT, read_timeout)
            
            logger.debug(f"Sending chat request (timeout: {timeout})")
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                # Update cache, model is now definitely loaded
                normalized_name = self.normalize_model_name(self.config.OLLAMA_MODEL_NAME)
                self.loaded_models_cache.add(normalized_name)
                self.last_model_check[normalized_name] = time.time()
                
                return response.json()
                
        except requests.Timeout:
            logger.error(f"Ollama request timed out (connect: {self.config.LLM_CONNECT_TIMEOUT}s, read: {read_timeout}s)")
            raise
        except requests.ConnectionError as e:
            logger.error(f"Connection error to Ollama server: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
    
    def _stream_response(self, response):
        """Handle streaming response from Ollama"""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield data
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding streaming response: {e}")
                    continue
    
    def preload_model(self, model_name=None):
        """
        Preload a model to avoid delay on first request
        
        Args:
            model_name: Name of model to preload (uses config model if not specified)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_name is None:
            model_name = self.config.OLLAMA_MODEL_NAME
        
        try:
            logger.info(f"Preloading model '{model_name}'...")
            
            # Send a simple request to trigger model loading
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    'model': model_name,
                    'messages': [{'role': 'user', 'content': 'test'}],
                    'stream': False
                },
                timeout=(self.config.LLM_CONNECT_TIMEOUT, 180)  # 3 minutes for loading
            )
            response.raise_for_status()
            
            # Update cache
            self.loaded_models_cache.add(model_name)
            self.last_model_check[model_name] = time.time()
            
            logger.info(f"✓ Model '{model_name}' preloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
            return False

class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self, config):
        self.config = config
        self.messages = []
        self.last_activity = datetime.now()
        self.load_context()
    
    def load_context(self):
        """Load conversation context from file"""
        try:
            if os.path.exists(self.config.CONTEXT_FILE):
                with open(self.config.CONTEXT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
                    
                    # Parse last activity timestamp
                    last_activity_str = data.get('last_activity')
                    if last_activity_str:
                        self.last_activity = datetime.fromisoformat(last_activity_str)
                    
                    # Check if context has expired
                    if self._is_context_expired():
                        logger.info("Context expired, starting fresh conversation")
                        self.clear_context()
                    else:
                        logger.info(f"Loaded {len(self.messages)} messages from context")
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            self.clear_context()
    
    def save_context(self):
        """Save conversation context to file"""
        try:
            data = {
                'messages': self.messages,
                'last_activity': self.last_activity.isoformat()
            }
            with open(self.config.CONTEXT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Context saved successfully")
        except Exception as e:
            logger.error(f"Error saving context: {e}")
    
    def add_message(self, role, content):
        """Add a message to the conversation context"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Trim context if it gets too long
        self._trim_context()
        
        # Save context after adding message
        self.save_context()
        
        logger.debug(f"Added {role} message to context")
    
    def get_messages_for_ai(self):
        """Get messages formatted for AI model"""
        # Convert to format expected by Ollama
        ai_messages = []
        for msg in self.messages:
            ai_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        return ai_messages
    
    def _trim_context(self):
        """Trim context to keep only recent messages"""
        if len(self.messages) > self.config.MAX_CONTEXT_MESSAGES:
            # Keep the most recent messages
            excess = len(self.messages) - self.config.MAX_CONTEXT_MESSAGES
            self.messages = self.messages[excess:]
            logger.debug(f"Trimmed {excess} messages from context")
    
    def _is_context_expired(self):
        """Check if context has expired due to inactivity"""
        time_since_activity = datetime.now() - self.last_activity
        return time_since_activity > timedelta(minutes=self.config.CONTEXT_TIMEOUT_MINUTES)
    
    def clear_context(self):
        """Clear conversation context"""
        self.messages = []
        self.last_activity = datetime.now()
        
        # Remove context file
        if os.path.exists(self.config.CONTEXT_FILE):
            try:
                os.remove(self.config.CONTEXT_FILE)
                logger.info("Context file removed")
            except Exception as e:
                logger.error(f"Error removing context file: {e}")
        
        logger.info("Conversation context cleared")
    
    def get_context_summary(self):
        """Get a summary of current context"""
        if not self.messages:
            return "No conversation history"
        
        user_messages = len([m for m in self.messages if m['role'] == 'user'])
        ai_messages = len([m for m in self.messages if m['role'] == 'assistant'])
        
        time_since_activity = datetime.now() - self.last_activity
        minutes_ago = int(time_since_activity.total_seconds() / 60)
        
        return f"Context: {user_messages} user messages, {ai_messages} AI responses, last activity {minutes_ago} minutes ago"

class TypingSound:
    """Class to handle typing sound generation during processing"""    
    def __init__(self, config):
        self.config = config
        self.stop_typing = threading.Event()
        self.typing_thread = None
        self.pyaudio_instance = None
    
    def _generate_tone(self, frequency, duration, sample_rate=44100, volume=None):
        """Generate a single tone with smooth fade in/out"""
        if volume is None:
            volume = self.config.THINKING_SOUND_VOLUME * 0.15  # Scale down for individual tones
        
        frames = int(duration * sample_rate)
        arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
        
        # Smooth fade in/out to avoid clicks
        fade_frames = int(0.02 * sample_rate)  # 20ms fade
        if fade_frames > 0:
            arr[:fade_frames] *= np.linspace(0, 1, fade_frames)
            arr[-fade_frames:] *= np.linspace(1, 0, fade_frames)
        
        return (arr * volume * 32767).astype(np.int16)
    
    def _generate_melody(self, sample_rate=44100):
        """Generate a thinking melody"""
        # A gentle, thoughtful arpeggio pattern in A minor
        melody_notes = [
            440.00,  # A4
            523.25,  # C5
            659.25,  # E5
            523.25,  # C5
            440.00,  # A4
            329.63,  # E4
            440.00,  # A4
            523.25,  # C5
        ]
        
        note_duration = 0.18  # Slightly faster, more fluid
        melody_audio = np.array([], dtype=np.int16)
        
        for i, freq in enumerate(melody_notes):
            # Gentle volume swell
            volume = self.config.THINKING_SOUND_VOLUME * (0.8 + (0.2 * np.sin(i * 0.7)))
            
            note = self._generate_tone(freq, note_duration, sample_rate, volume * 0.12)
            melody_audio = np.concatenate([melody_audio, note])
            
            # Very short gap for smooth flow
            if i < len(melody_notes) - 1:
                gap_frames = int(0.03 * sample_rate)
                gap = np.zeros(gap_frames, dtype=np.int16)
                melody_audio = np.concatenate([melody_audio, gap])
        
        # Pause before loop
        end_pause_frames = int(0.6 * sample_rate)
        end_pause = np.zeros(end_pause_frames, dtype=np.int16)
        melody_audio = np.concatenate([melody_audio, end_pause])
        
        return melody_audio
    
    def _typing_loop(self):
        """Background thread function for playing the thinking melody"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Generate the thinking melody
            melody = self._generate_melody()
            
            # Open audio stream for playback
            stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                output=True,
                output_device_index=self.config.OUTPUT_DEVICE,
                frames_per_buffer=1024
            )
            
            # Calculate chunk size for smooth playback
            chunk_size = 1024
            
            while not self.stop_typing.is_set():
                # Play melody in chunks
                for i in range(0, len(melody), chunk_size):
                    if self.stop_typing.is_set():
                        break
                    
                    chunk = melody[i:i + chunk_size]
                    
                    # Pad chunk if necessary
                    if len(chunk) < chunk_size:
                        padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
                        chunk = np.concatenate([chunk, padding])
                    
                    stream.write(chunk.tobytes())
                    
                    # Small delay to prevent overwhelming the audio system
                    time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error in typing sound thread: {e}")
        finally:
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
    
    def start(self):
        """Start the typing sound loop"""
        # Check if thinking sound is enabled
        if not self.config.THINKING_SOUND_ENABLED:
            logger.debug("Thinking sound is disabled in config")
            return
        
        if self.typing_thread and self.typing_thread.is_alive():
            return
        
        self.stop_typing.clear()
        self.typing_thread = threading.Thread(target=self._typing_loop, daemon=True)
        self.typing_thread.start()
        logger.debug("Typing sound started")
    
    def stop(self):
        """Stop the typing sound loop"""
        if self.typing_thread and self.typing_thread.is_alive():
            self.stop_typing.set()
            self.typing_thread.join(timeout=1.0)
            logger.debug("Typing sound stopped")

class HamRadioAI:
    def __init__(self):
        self.config = AIConfig()
        self.whisper_model = None
        self.pyaudio_instance = None
        self.typing_sound = TypingSound(self.config)
        self.context = ConversationContext(self.config)
        self._initialize_models()
        self.piper_voice = None
        self._load_piper_voice()
        self.ollama_client = OllamaClient(self.config)
    
        # Test connection on startup
        if not self.ollama_client.test_connection():
            logger.warning("⚠ Ollama server not accessible - AI will fail until server is available")
        else:
            # List available models
            models = self.ollama_client.list_models()
            logger.info("Available models:\n\t" + "\n\t".join(models))
            if self.config.OLLAMA_MODEL_NAME not in models:
                # Try with :latest suffix
                normalized_name = self.ollama_client.normalize_model_name(self.config.OLLAMA_MODEL_NAME)
                if normalized_name not in models:
                    logger.warning(f"⚠ Model '{self.config.OLLAMA_MODEL_NAME}' not found on server")
            
            # Check if model is loaded
            is_loaded = self.ollama_client.is_model_loaded(self.config.OLLAMA_MODEL_NAME)
            
            # Optionally preload model
            if self.config.PRELOAD_MODEL and not is_loaded:
                logger.info("Preloading model (this may take a minute)...")
                self.ollama_client.preload_model()

    def _initialize_models(self):
        """Initialize AI models and audio system"""
        try:
            logger.info("Initializing Whisper model...")
            self.whisper_model = WhisperModel(
                self.config.WHISPER_MODEL_SIZE, 
                device="cpu", 
                compute_type="int8"
            )
            logger.info("Whisper model initialized successfully")
            
            self.pyaudio_instance = pyaudio.PyAudio()
            logger.info("PyAudio initialized successfully")
            
            # Log context status
            logger.info(self.context.get_context_summary())
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _process_audio_channels(self, audio_np):
        """Process audio channels based on configuration"""
        output_channels = self.config.CHANNELS  # Default to input channels
        
        if self.config.CHANNELS == 2:
            # Reshape to separate channels
            audio_np = audio_np.reshape(-1, 2)
            
            if self.config.INPUT_CHANNEL == 'left':
                audio_np = audio_np[:, 0]  # Keep only left channel
                output_channels = 1
            elif self.config.INPUT_CHANNEL == 'right':
                audio_np = audio_np[:, 1]  # Keep only right channel
                output_channels = 1
            elif self.config.INPUT_CHANNEL == 'mono':
                audio_np = np.mean(audio_np, axis=1).astype(np.int16)  # Average both
                output_channels = 1
            elif self.config.INPUT_CHANNEL == 'both':
                # Keep stereo - flatten back
                audio_np = audio_np.flatten()
                output_channels = 2
            else:
                logger.warning(f"Unknown input_channel: {self.config.INPUT_CHANNEL}, using both")
                audio_np = audio_np.flatten()
                output_channels = 2
        
        return audio_np, output_channels
    
    @contextmanager
    def audio_stream(self):
        """Context manager for audio stream"""
        stream = None
        try:
            stream = self.pyaudio_instance.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                input=True,
                input_device_index=self.config.INPUT_DEVICE,
                frames_per_buffer=self.config.CHUNK
            )
            yield stream
        except Exception as e:
            logger.error(f"Error with audio stream: {e}")
            raise
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def _check_cancel_flag(self):
        """Check if user has requested cancellation"""
        return os.path.exists(self.config.CANCEL_FILE)
    
    def _create_ready_flag(self):
        """Create the ready flag to signal main repeater"""
        try:
            with open(self.config.READY_FILE, "w") as f:
                f.write("ready")
            logger.info("Ready flag created")
        except Exception as e:
            logger.error(f"Failed to create ready flag: {e}")
    
    def _cleanup_files(self):
        """Clean up temporary files"""
        files_to_remove = [
            self.config.READY_FILE,
            self.config.OUTPUT_FILE
        ]
        
        for file in files_to_remove:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logger.debug(f"Removed {file}")
                except OSError as e:
                    logger.warning(f"Could not remove {file}: {e}")
    
    def record_audio(self, timeout=None):
        """Record audio from microphone until silence or timeout"""
        if timeout is None:
            timeout = self.config.DEFAULT_TIMEOUT
        
        logger.info("Starting audio recording...")
        
        # Create ready flag to signal the main repeater
        self._create_ready_flag()
        
        frames = []
        silent_chunks = 0
        silence_limit = int(self.config.SILENCE_LIMIT_SECONDS * self.config.RATE / self.config.CHUNK)
        speech_chunks = 0
        min_speech_chunks = int(self.config.MIN_TALKING * self.config.RATE / self.config.CHUNK)
        speaking_started = False
        last_audio_time = time.time()
        
        # IMPORTANT: Record in mono
        input_channels = 1
        
        try:
            # Open stream directly
            stream = self.pyaudio_instance.open(
                format=self.config.FORMAT,
                channels=input_channels,  # Use 1 channel
                rate=self.config.RATE,
                input=True,
                input_device_index=self.config.INPUT_DEVICE,
                frames_per_buffer=self.config.CHUNK
            )
            
            logger.info("Waiting for speech...")
            
            while True:
                # Check for cancellation
                if self._check_cancel_flag():
                    logger.info("AI mode cancelled by user")
                    stream.stop_stream()
                    stream.close()
                    return None
                
                # Check for timeout
                if time.time() - last_audio_time > timeout:
                    logger.info("Recording timeout reached")
                    stream.stop_stream()
                    stream.close()
                    return None
                
                # Read audio data
                try:
                    data = stream.read(self.config.CHUNK, exception_on_overflow=False)
                    audio_np = np.frombuffer(data, dtype=np.int16)
                    
                except Exception as e:
                    logger.error(f"Error reading audio data: {e}")
                    continue
                
                # Check audio level
                if np.max(np.abs(audio_np)) < self.config.THRESHOLD:
                    silent_chunks += 1
                    speech_chunks = 0

                    if not speaking_started:
                        frames.clear()

                else:
                    speech_chunks += 1
                    silent_chunks = 0

                    # Only start speaking after minimum duration
                    if not speaking_started and speech_chunks >= min_speech_chunks:
                        speaking_started = True

                    if speaking_started:
                        frames.append(data)
                        last_audio_time = time.time()
                
                # Check if we should stop recording
                if speaking_started and silent_chunks > silence_limit:
                    logger.info("Speech ended, stopping recording")
                    break
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save recorded audio
            if frames:
                return self._save_audio_frames(frames, channels=1)  # Pass channels parameter
            else:
                logger.warning("No audio frames recorded")
                return None
                
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            # Ensure stream is closed on error
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            return None

    def _save_audio_frames(self, frames, channels=1):
        """Save audio frames to WAV file"""
        try:
            with wave.open(self.config.OUTPUT_FILE, 'wb') as wf:
                wf.setnchannels(channels)  # Use the passed channels parameter
                wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.config.FORMAT))
                wf.setframerate(self.config.RATE)
                wf.writeframes(b''.join(frames))
            
            logger.info(f"Audio saved to {self.config.OUTPUT_FILE}")
            return self.config.OUTPUT_FILE
            
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using Whisper"""
        if not os.path.exists(audio_file):
            logger.error(f"Audio file {audio_file} not found")
            return None

        try:
            logger.info("Transcribing audio...")
            
            # Start typing sound during transcription
            self.typing_sound.start()
            
            segments, info = self.whisper_model.transcribe(
                audio_file, 
                language="el"  # Greek language
            )

            if info.duration < 2.0:
                logger.info(f"Audio ignored: duration is shorter than 2 seconds.")
                return None
            
            result = " ".join([segment.text.strip() for segment in segments])
            
            if result.strip():
                logger.info(f"Transcription: {result}")
                return result.strip()
            else:
                logger.warning("Empty transcription result")
                return None
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
        finally:
            # Note: Don't stop typing sound here, continue through AI response generation
            pass
    
    def generate_response(self, prompt):
        """Generate AI response with Ollama"""
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided")
            return None
        
        try:
            logger.info(f"Generating AI response... (server: {self.config.get_ollama_base_url()})")
            
            # Add user message to context
            self.context.add_message("user", prompt)
            
            # Get conversation history for AI
            system_prompt = self.config.load_system_prompt()
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.context.get_messages_for_ai())
            
            # Call Ollama with automatic timeout adjustment
            try:
                response = self.ollama_client.chat(
                    messages, 
                    stream=False,
                    auto_adjust_timeout=self.config.AUTO_ADJUST_TIMEOUT
                )
                ai_text = response['message']['content'].strip()
                
            except requests.Timeout:
                timeout_used = self.ollama_client.get_dynamic_timeout(self.config.OLLAMA_MODEL_NAME)
                logger.error(f"LLM request timed out after {timeout_used}s")
                fallback_msg = "Συγγνώμη, μου πήρε πολύ χρόνο να απαντήσω. Παρακαλώ δοκιμάστε ξανά."
                self.context.add_message("assistant", fallback_msg)
                return fallback_msg
            
            except requests.ConnectionError:
                logger.error(f"Cannot connect to Ollama server at {self.config.get_ollama_base_url()}")
                fallback_msg = "Συγγνώμη, δεν μπορώ να συνδεθώ με τον διακομιστή. Επικοινωνήστε με τον διαχειριστή."
                self.context.add_message("assistant", fallback_msg)
                return fallback_msg
            
            except requests.RequestException as e:
                logger.error(f"Error communicating with Ollama: {e}")
                fallback_msg = "Συγγνώμη, παρουσιάστηκε σφάλμα κατά την επικοινωνία με τον διακομιστή."
                self.context.add_message("assistant", fallback_msg)
                return fallback_msg
            
            if ai_text:
                logger.info(f"AI Response: {ai_text}")
                self.context.add_message("assistant", ai_text)
                return ai_text
            else:
                logger.warning("Empty AI response")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error generating AI response: {e}")
            return None
        finally:
            self.typing_sound.stop()
            
    def play_audio(self, filename):
        """Play a WAV file through the audio output"""
        try:
            # Clamp volume to valid range
            volume = max(0.0, min(5.0, self.config.OUTPUT_VOLUME))
        
            with wave.open(filename, 'rb') as wf:
                audio_stream = self.pyaudio_instance.open(
                    format=self.pyaudio_instance.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self.config.OUTPUT_DEVICE
                )
                
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                while data:
                    # Convert bytes to numpy array for volume adjustment
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    # Apply volume (with gain) and clip to prevent distortion
                    audio_data = np.clip(audio_data * volume, -32768, 32767).astype(np.int16)
                    # Write adjusted audio
                    audio_stream.write(audio_data.tobytes())
                    data = wf.readframes(chunk_size)
                
                audio_stream.close()
                
        except FileNotFoundError:
            logger.error(f"Audio file {filename} not found!")
        except Exception as e:
            logger.error(f"Error playing audio file {filename}: {e}")
    
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

        sentence_endings = r'[.!?;]\s+'
        sentences = re.split(f'({sentence_endings})', text)

        result = []
        buffer = None  # holds a one-word sentence

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i + 1]).strip()
            else:
                sentence = sentences[i].strip()

            if not sentence:
                continue

            word_count = len(sentence.split())

            # If this is a one-word sentence, buffer it
            if word_count == 1:
                buffer = sentence
                continue

            # If we have a buffered one-word sentence, prepend it
            if buffer:
                sentence = f"{buffer} {sentence}"
                buffer = None

            result.append(sentence)

        # If text ends with a one-word sentence, keep it
        if buffer:
            result.append(buffer)

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
        
    def speak_text(self, text):
        """Convert text to speech using Piper with sentence splitting"""
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return False
        
        if not self.piper_voice:
            logger.error("Piper voice not loaded")
            return False
        
        try:
            logger.info("Converting text to speech with Piper...")
            
            # Ensure typing sound is stopped before TTS
            self.typing_sound.stop()
            
            # Split text into sentences for better flow
            sentences = self._split_sentences(text)
            logger.info(f"Speaking {len(sentences)} sentence(s)")
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                temp_file = f"audio/temp/piper_ai_temp_{i}.wav"
                
                # Generate speech to WAV file
                with wave.open(temp_file, "wb") as wav_file:
                    self.piper_voice.synthesize_wav(sentence, wav_file)
                
                # Play the generated audio using our play_audio method
                self.play_audio(temp_file)
                
                # Small pause between sentences (except for the last one)
                if i < len(sentences) - 1:
                    time.sleep(0.1)  # 100ms pause between sentences for AI responses
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logger.info("Text-to-speech completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}")
            return False
    
    def handle_special_commands(self, text):
        """Handle special commands like clearing context"""
        text_lower = text.lower().strip()
        
        # Check for context clear commands in Greek and English
        clear_commands = [
            "σβήσε το ιστορικό", "καθάρισε το ιστορικό", "ξεκίνα από την αρχή",
            "clear context", "clear history", "start fresh", "new conversation"
        ]
        
        for command in clear_commands:
            if command in text_lower:
                self.context.clear_context()
                logger.info("Context cleared by user command")
                return "Το ιστορικό της συζήτησης έχει σβηστεί. Μπορούμε να ξεκινήσουμε μια νέα συζήτηση."
        
        # Check for context status commands
        status_commands = [
            "πόσα μηνύματα", "τι θυμάσαι", "ιστορικό συζήτησης",
            "conversation status", "what do you remember", "context status"
        ]
        
        for command in status_commands:
            if command in text_lower:
                summary = self.context.get_context_summary()
                if len(self.context.messages) == 0:
                    return "Δεν έχουμε ιστορικό συζήτησης ακόμα. Αυτό είναι το πρώτο μας μήνυμα."
                else:
                    user_msgs = len([m for m in self.context.messages if m['role'] == 'user'])
                    ai_msgs = len([m for m in self.context.messages if m['role'] == 'assistant'])
                    return f"Έχουμε ανταλλάξει {user_msgs} μηνύματα από εσένα και {ai_msgs} απαντήσεις από εμένα σε αυτή τη συζήτηση."
        
        return None  # No special command found
    
    def run_ai_session(self):
        """Run a single AI interaction session"""
        # Record audio
        audio_file = self.record_audio()
        if not audio_file:
            logger.info("No audio recorded, ending session")
            return False
        
        # Transcribe audio (starts typing sound)
        transcribed_text = self.transcribe_audio(audio_file)
        if not transcribed_text:
            logger.warning("No transcription available")
            # Stop typing sound if transcription failed
            self.typing_sound.stop()
            return True  # Continue session
        
        # Check for special commands first
        special_response = self.handle_special_commands(transcribed_text)
        if special_response:
            # Stop typing sound for special commands
            self.typing_sound.stop()
            
            # Speak the special response
            success = self.speak_text(special_response)
            if not success:
                logger.warning("Failed to speak special command response")
            return True  # Continue session
        
        # Generate AI response (continues typing sound, then stops it)
        ai_response = self.generate_response(transcribed_text)
        if not ai_response:
            logger.warning("No AI response generated")
            # Ensure typing sound is stopped
            self.typing_sound.stop()
            return True  # Continue session
        else:
            ai_response = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()
        
        # Speak the response (typing sound should already be stopped)
        time.sleep(0.5)
        success = self.speak_text(ai_response)
        if not success:
            logger.warning("Failed to speak AI response")
        
        return True  # Continue session
    
    def run(self):
        """Main AI mode loop"""
        logger.info("Starting Ham Radio AI mode...")
        
        try:
            while True:
                # Check for cancellation before starting new session
                if self._check_cancel_flag():
                    logger.info("AI mode cancelled by user")
                    break
                
                # Run AI interaction session
                if not self.run_ai_session():
                    break
                
        except KeyboardInterrupt:
            logger.info("AI mode interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in AI mode: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of AI mode"""
        logger.info("Shutting down AI mode...")
        
        # Stop typing sound
        self.typing_sound.stop()
        
        # Save final context state
        self.context.save_context()
        
        # Clean up files
        self._cleanup_files()
        
        # Close PyAudio
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        logger.info("AI mode shutdown complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ham Radio AI Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        default='aimode_config.ini',
        help='Path to configuration file (default: aimode_config.ini)'
    )
    
    parser.add_argument(
        '--no-thinking-sound',
        action='store_true',
        help='Disable thinking sound during processing'
    )
    
    parser.add_argument(
        '--model',
        help='Override Ollama model from config'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        help='Override AI temperature from config'
    )
    
    args = parser.parse_args()
    
    try:
        ai = HamRadioAI()
        
        # Apply command-line overrides
        if args.no_thinking_sound:
            ai.config.THINKING_SOUND_ENABLED = False
        
        if args.model:
            ai.config.OLLAMA_MODEL_NAME = args.model
            logger.info(f"Using model: {args.model}")
        
        if args.temperature is not None:
            ai.config.TEMPERATURE = args.temperature
            logger.info(f"Using temperature: {args.temperature}")
        
        ai.run()
        
    except Exception as e:
        logger.error(f"Fatal error in AI mode: {e}")
        if 'ai' in locals():
            ai.shutdown()

if __name__ == "__main__":
    main()

