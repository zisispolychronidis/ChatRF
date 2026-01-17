"""
Callsign Lookup Module for ChatRF

This module looks up amateur radio callsigns using voice recognition
when DTMF command '5' is pressed.
"""

import os
import re
from modules.base import DTMFModule


class CallsignLookupModule(DTMFModule):
    """
    Looks up callsign information from the RadioID database.
    """
    
    # Module metadata
    name = "Callsign Lookup"
    version = "1.0.0"
    description = "Callsign lookup"
    
    # DTMF configuration
    dtmf_command = '5'
    requires_rate_limit = True
    
    def handle_command(self):
        """Execute when DTMF '5' is received."""

        # Schedule the lookup to be performed by the main loop
        flag_name = self.repeater.module_manager._create_flag_name(self.name)
        self.repeater.module_manager.set_flag(flag_name)
    
    def execute(self):
        """Standardized execution method called by ModuleManager."""

        self.do_callsign_lookup()
    
    def do_callsign_lookup(self):
        """Callsign lookup logic"""
        try:
            # Play the prompt asking for callsign
            self.repeater.play_audio(self.config.CALLSIGN_PROMPT_FILE)
            
            # Record audio
            audio_file = self.repeater.record_callsign_audio(duration=8)
            if not audio_file:
                raise Exception("Audio recording failed")

            # Transcribe with Whisper
            raw_transcription = self.repeater.transcribe_audio_whisper(audio_file)
            callsign_text = self.repeater.extract_callsign_from_text(raw_transcription)
            
            if not callsign_text:
                message = "Δεν κατάλαβα το διακριτικό. Παρακαλώ δοκιμάστε ξανά."
            else:
                # Look up in database
                info = self.repeater.lookup_callsign_in_db(callsign_text)
                if info:
                    name = f"{info['name']}".strip()
                    city = info['city']
                    country = info['country']
                    message = (
                        f"Το διακριτικό {' '.join(callsign_text)} ανήκει στον {name} "
                        f"από {city}, {country}."
                    )
                else:
                    message = f"Δεν βρέθηκε το διακριτικό {' '.join(callsign_text)}."

            self.logger.info(f"Speaking lookup result: {message}")
            self.repeater.speak_with_piper(message)
            
            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)

        except Exception as e:
            self.logger.error(f"Error during callsign lookup: {e}")
            # Play error message
            error_msg = "Σφάλμα κατά την αναζήτηση διακριτικού."
            self.repeater.speak_with_piper(error_msg)
    
    def can_execute(self):
        """Check if the command can execute."""
        # Only execute if not in AI mode
        if self.repeater.ai_mode_running:
            self.logger.info("Callsign lookup blocked - AI mode is active")
            return False
        
        return True
    
    def cleanup(self):
        """Called when module is unloaded or system shuts down"""
        self.logger.info("Callsign lookup module shutting down")
