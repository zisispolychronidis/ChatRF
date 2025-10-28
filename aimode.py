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
from pathlib import Path
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from contextlib import contextmanager
from piper import PiperVoice

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.FORMAT = pyaudio.paInt16  # This stays hardcoded
        self.CHANNELS = self.config.getint('Audio', 'channels', fallback=1)
        self.RATE = self.config.getint('Audio', 'rate', fallback=44100)
        self.CHUNK = self.config.getint('Audio', 'chunk', fallback=1024)
        self.THRESHOLD = self.config.getint('Audio', 'threshold', fallback=500)
        
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
        
        # Timing Settings
        self.DEFAULT_TIMEOUT = self.config.getint('Timing', 'default_timeout', fallback=60)
        self.SILENCE_LIMIT_SECONDS = self.config.getint('Timing', 'silence_limit_seconds', fallback=2)
        
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
            'rate': '44100',
            'chunk': '1024',
            'threshold': '500'
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
            'temperature': '0.2'
        }
        
        default_config['Timing'] = {
            'default_timeout': '60',
            'silence_limit_seconds': '2'
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
        speaking_started = False
        last_audio_time = time.time()
        
        try:
            with self.audio_stream() as stream:
                logger.info("Waiting for speech...")
                
                while True:
                    # Check for cancellation
                    if self._check_cancel_flag():
                        logger.info("AI mode cancelled by user")
                        return None
                    
                    # Check for timeout
                    if time.time() - last_audio_time > timeout:
                        logger.info("Recording timeout reached")
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
                        if not speaking_started:
                            frames.clear()  # Clear buffer if we haven't started speaking
                    else:
                        # Audio detected
                        speaking_started = True
                        silent_chunks = 0
                        frames.append(data)
                        last_audio_time = time.time()
                    
                    # Check if we should stop recording
                    if speaking_started and silent_chunks > silence_limit:
                        logger.info("Speech ended, stopping recording")
                        break
                
                # Save recorded audio
                if frames:
                    return self._save_audio_frames(frames)
                else:
                    logger.warning("No audio frames recorded")
                    return None
                    
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            return None
    
    def _save_audio_frames(self, frames):
        """Save audio frames to WAV file"""
        try:
            with wave.open(self.config.OUTPUT_FILE, 'wb') as wf:
                wf.setnchannels(self.config.CHANNELS)
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
        """Generate AI response using Ollama with conversation context"""
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided")
            return None
        
        try:
            logger.info("Generating AI response...")
            
            # Typing sound should already be running from transcription
            
            # Add user message to context
            self.context.add_message("user", prompt)
            
            # Get conversation history for AI
            system_prompt = self.config.load_system_prompt()
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.context.get_messages_for_ai())
            
            response = ollama.chat(
                model=self.config.OLLAMA_MODEL_NAME,
                messages=messages,
                options={"temperature": self.config.TEMPERATURE}
            )
            
            ai_text = response["message"]["content"].strip()
            
            if ai_text:
                logger.info(f"AI Response: {ai_text}")
                
                # Add AI response to context
                self.context.add_message("assistant", ai_text)
                
                return ai_text
            else:
                logger.warning("Empty AI response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None
        finally:
            # Stop typing sound after AI response is generated
            self.typing_sound.stop()
            
    def play_audio(self, filename):
        """Play a WAV file through the audio output"""
        try:
            with wave.open(filename, 'rb') as wf:
                audio_stream = self.pyaudio_instance.open(
                    format=self.pyaudio_instance.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                while data:
                    audio_stream.write(data)
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
        
        # Greek and English sentence endings
        sentence_endings = r'[.!?;]'
        
        # Split on sentence endings but keep the punctuation
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
                    time.sleep(0.4)  # 400ms pause between sentences for AI responses
                
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
        
        # Speak the response (typing sound should already be stopped)
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
