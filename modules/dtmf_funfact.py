"""
Fun Fact Module for ChatRF

This module plays a random fun fact from a text file when DTMF command '4' is pressed.
"""

import os
import random
from modules.base import DTMFModule


class FunFactModule(DTMFModule):
    """
    Plays a random fun fact when triggered via DTMF.
    
    Configuration:
        - Reads facts from the file specified in config.FUN_FACTS_FILE
        - One fact per line in the text file
    """
    
    # Module metadata
    name = "Fun Fact"
    version = "1.0.0"
    description = "Random fun fact"
    
    # DTMF configuration
    dtmf_command = '4'
    requires_rate_limit = True
    
    def initialize(self):
        """
        Called when module is loaded.
        """
        self.fact_file = self.config.FUN_FACTS_FILE
        
        # Check if the fact file exists
        if not os.path.exists(self.fact_file):
            self.logger.warning(f"Fun facts file not found: {self.fact_file}")
        else:
            # Count how many facts we have
            with open(self.fact_file, "r", encoding="utf-8") as f:
                fact_count = len([line for line in f if line.strip()])
            self.logger.info(f"Loaded {fact_count} fun facts from {self.fact_file}")
    
    def handle_command(self):
        """Execute when DTMF '4' is received."""

        # Schedule the fact to be played by the main loop
        flag_name = self.repeater.module_manager._create_flag_name(self.name)
        self.repeater.module_manager.set_flag(flag_name)

    def execute(self):
        """
        This is the standardized execution method called by ModuleManager.
        All DTMF modules must implement this method.
        """
        self.play_fact()
    
    def get_random_fact(self):
        """
        Get a random fact from the file.
        
        Returns:
            str: A random fact, or None if file is empty/missing
        """
        try:
            if not os.path.exists(self.fact_file):
                self.logger.error(f"Fact file not found: {self.fact_file}")
                return None
            
            with open(self.fact_file, "r", encoding="utf-8") as f:
                facts = [line.strip() for line in f if line.strip()]
            
            if not facts:
                self.logger.warning("Fun facts file is empty")
                return None
            
            return random.choice(facts)
            
        except Exception as e:
            self.logger.error(f"Error reading fun facts: {e}")
            return None
    
    def play_fact(self):
        """
        Actually play the fun fact using TTS.
        This is called by the main loop when the flag is True.
        """
        try:
            fact = self.get_random_fact()
            
            if fact:
                self.logger.info(f"Speaking fun fact: {fact}")
                self.repeater.speak_with_piper(f"Fun fact: {fact}")
            else:
                self.logger.warning("No fact available to play")
                
        except Exception as e:
            self.logger.error(f"Error during fun fact playback: {e}")
    
    def can_execute(self):
        """
        Check if the command can execute.
        """
        # Only execute if not in AI mode
        if self.repeater.ai_mode_running:
            self.logger.info("Fun fact blocked - AI mode is active")
            return False
        
        return True
    
    def cleanup(self):
        """Called when module is unloaded or system shuts down"""
        self.logger.info("Fun fact module shutting down")