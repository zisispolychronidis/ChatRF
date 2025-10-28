import pyaudio
import numpy as np
import threading
import time
import RPi.GPIO as GPIO
import logging

logger = logging.getLogger(__name__)

class SimpleAudioPTT:
    """Simple PTT controller"""
    
    def __init__(self, ptt_pin=18, ptt_tail=0.5, 
                 silence_threshold=100, monitor_device=None):
        """
        Simple PTT based on monitoring audio output levels
        
        Args:
            ptt_pin: GPIO pin for PTT
            ptt_tail: Delay after silence before PTT off
            silence_threshold: Audio level threshold for silence
            monitor_device: Audio device to monitor (None = default)
        """
        self.ptt_pin = ptt_pin
        self.ptt_tail = ptt_tail
        self.silence_threshold = silence_threshold
        self.monitor_device = monitor_device
        
        self.ptt_active = False
        self.monitoring = False
        self.last_audio_time = 0
        self.ptt_timer = None
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.ptt_pin, GPIO.OUT)
        GPIO.output(self.ptt_pin, GPIO.LOW)
        
        # Setup PyAudio for monitoring
        self.p = pyaudio.PyAudio()
        
        logger.info(f"Simple Audio PTT initialized on GPIO {ptt_pin}")
    
    def _find_output_device(self):
        """Find the default output device for monitoring"""
        try:
            default_output_index = self.p.get_default_output_device_info()['index']
            
            # Look for a corresponding input device (for monitoring)
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if (device_info['maxInputChannels'] > 0 and 
                    'monitor' in device_info['name'].lower()):
                    return i
            
            # Fallback to default input
            return self.p.get_default_input_device_info()['index']
            
        except Exception as e:
            logger.warning(f"Could not find monitor device: {e}")
            return None
    
    def _ptt_on(self):
        """Turn PTT on"""
        if not self.ptt_active:
            GPIO.output(self.ptt_pin, GPIO.HIGH)
            self.ptt_active = True
            logger.debug("PTT ON")
    
    def _ptt_off(self):
        """Turn PTT off"""
        if self.ptt_active:
            GPIO.output(self.ptt_pin, GPIO.LOW)
            self.ptt_active = False
            logger.debug("PTT OFF")
    
    def _monitor_audio(self):
        """Monitor audio levels and control PTT"""
        
        device_index = self.monitor_device or self._find_output_device()
        if device_index is None:
            logger.error("No suitable audio device found for monitoring")
            return
        
        try:
            # Open monitoring stream
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            
            logger.info(f"Monitoring audio device {device_index}")
            
            silence_time = 0
            
            while self.monitoring:
                try:
                    # Read audio data
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate audio level
                    audio_level = np.max(np.abs(audio_data))
                    
                    if audio_level > self.silence_threshold:
                        # Audio detected
                        self.last_audio_time = time.time()
                        silence_time = 0
                        
                        # Turn PTT on
                        if not self.ptt_active:
                            self._ptt_on()
                        
                        # Cancel any pending PTT off timer
                        if self.ptt_timer:
                            self.ptt_timer.cancel()
                            self.ptt_timer = None
                    
                    else:
                        # Silence detected
                        if self.ptt_active:
                            current_time = time.time()
                            silence_time = current_time - self.last_audio_time
                            
                            # If silence long enough, schedule PTT off
                            if silence_time > 0.1 and not self.ptt_timer:  # 100ms of silence
                                self.ptt_timer = threading.Timer(
                                    self.ptt_tail,
                                    self._ptt_off
                                )
                                self.ptt_timer.start()
                
                except Exception as e:
                    logger.debug(f"Audio monitoring error: {e}")
                    time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Failed to start audio monitoring: {e}")
    
    def start_monitoring(self):
        """Start audio monitoring and PTT control"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_audio,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Audio PTT monitoring started")
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.monitoring = False
        
        if self.ptt_timer:
            self.ptt_timer.cancel()
        
        self._ptt_off()
        
        logger.info("Audio PTT monitoring stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        self.p.terminate()
        GPIO.cleanup()

# Integration with repeater script
def ptt_for_repeater(repeater_instance, ptt_pin=18):
    """
    Add automatic PTT to existing repeater instance
    This will trigger PTT for ANY (system-wide) audio output
    """
    
    # Create PTT controller
    ptt_controller = SimpleAudioPTT(ptt_pin=ptt_pin)
    
    # Start monitoring
    ptt_controller.start_monitoring()
    
    # Store reference for cleanup
    repeater_instance.ptt_controller = ptt_controller
    
    # Modify shutdown to cleanup PTT
    original_shutdown = repeater_instance.shutdown
    
    def enhanced_shutdown(self):
        if hasattr(self, 'ptt_controller'):
            self.ptt_controller.cleanup()
        original_shutdown()
    
    # Replace shutdown method
    repeater_instance.shutdown = enhanced_shutdown.__get__(
        repeater_instance, 
        type(repeater_instance)
    )
    
    logger.info("Automatic PTT added to repeater")
    return repeater_instance

# Standalone usage
if __name__ == "__main__":
    # Create and start PTT monitor
    ptt = SimpleAudioPTT(ptt_pin=18)
    ptt.start_monitoring()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ptt.cleanup()
        print("PTT monitor stopped")