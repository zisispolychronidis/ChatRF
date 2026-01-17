"""
ChatRF Module System - Base Classes

This module defines the base classes for all ChatRF plugin modules.
All custom modules should inherit from one of these base classes.
"""

import logging
from abc import ABC, abstractmethod
import threading


class BaseChatRFModule(ABC):
    """
    Base class for all ChatRF modules.
    
    Attributes:
        name (str): Name of the module
        version (str): Module version
        enabled (bool): Whether this module is active
        description (str): Brief description of what the module does
    """
    
    name = "Base Module"
    version = "1.0.0"
    enabled = True
    description = "Base module class"
    
    def __init__(self, repeater):
        """
        Initialize the module.
        
        Args:
            repeater: Reference to the main HamRepeater instance
        """
        self.repeater = repeater
        self.config = repeater.config
        self.logger = logging.getLogger(f"Module.{self.name}")
    
    def initialize(self):
        """
        Called when the module is first loaded.
        Override this to perform any setup tasks.
        """
        pass
    
    def cleanup(self):
        """
        Called when the module is being unloaded or system is shutting down.
        Override this to perform cleanup tasks.
        """
        pass


class DTMFModule(BaseChatRFModule):
    """
    Base class for DTMF command modules.
    
    These modules are triggered when a specific DTMF tone is detected.
    
    Attributes:
        dtmf_command (str): The DTMF command that triggers this module (e.g., '1', '2', '*', '#')
        requires_rate_limit (bool): Whether to apply rate limiting to this command
        description (str): What this command does (shown in menu)
    """
    
    dtmf_command = None  # Must be set by subclass
    requires_rate_limit = True
    description = "DTMF command module"
    
    def __init__(self, repeater):
        super().__init__(repeater)
        
        if self.dtmf_command is None:
            raise ValueError(f"Module {self.name} must set dtmf_command attribute")
    
    @abstractmethod
    def handle_command(self):
        """
        Execute when the DTMF command is received.
        This method must be implemented by subclasses.
        
        This method should NOT block - if you need to do something time-consuming,
        set a flag on the repeater object and let the main loop handle it.
        """
        pass

    @abstractmethod
    def execute(self):
        """
        Execute the actual module functionality.
        This is called by the main loop when the module's flag is set.
        """
        pass
    
    def can_execute(self):
        """
        Check if this command can execute right now.
        Override this to add custom conditions (e.g., only during certain hours).
        
        Returns:
            bool: True if command can execute, False otherwise
        """
        return True


class PeriodicModule(BaseChatRFModule):
    """
    Base class for periodic/scheduled modules.
    
    These modules run automatically on a schedule.
    
    Attributes:
        interval_seconds (int): How often to run this module (in seconds)
        run_immediately (bool): Whether to run once immediately on startup
        wait_for_silence (bool): Whether to wait for radio silence before executing
    """
    
    interval_seconds = 600  # Default: 10 minutes
    run_immediately = False
    wait_for_silence = True
    description = "Periodic task module"
    
    def __init__(self, repeater):
        super().__init__(repeater)
        self._thread = None
        self._stop_event = threading.Event()
    
    @abstractmethod
    def execute(self):
        """
        Execute the periodic task.
        This method must be implemented by subclasses.
        
        This is called automatically on the schedule defined by interval_seconds.
        """
        pass
    
    def start(self):
        """Start the periodic execution thread"""
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning(f"Module {self.name} already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Started periodic module: {self.name}")
    
    def stop(self):
        """Stop the periodic execution thread"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.logger.info(f"Stopped periodic module: {self.name}")
    
    def _run_loop(self):
        """Internal method - runs the periodic loop"""
        import time
        
        # Run immediately if requested
        if self.run_immediately:
            self._safe_execute()
        
        # Main periodic loop
        while not self._stop_event.is_set():
            # Wait for the interval (but check stop event frequently)
            for _ in range(self.interval_seconds):
                if self._stop_event.is_set():
                    return
                time.sleep(1)
            
            # Execute the task
            self._safe_execute()
    
    def _safe_execute(self):
        """Execute with error handling and optional silence waiting"""
        try:
            # Wait for silence if required
            if self.wait_for_silence:
                while self.repeater.talking:
                    if self._stop_event.is_set():
                        return
                    import time
                    time.sleep(0.5)
            
            # Execute the module
            self.execute()
            
        except Exception as e:
            self.logger.error(f"Error executing periodic module {self.name}: {e}")


class BackgroundServiceModule(BaseChatRFModule):
    """
    Base class for background service modules.
    
    These modules run continuously in their own thread, like the DTMF listener.
    
    Attributes:
        description (str): What this service does
    """
    
    description = "Background service module"
    
    def __init__(self, repeater):
        super().__init__(repeater)
        self._thread = None
        self._stop_event = threading.Event()
    
    @abstractmethod
    def run(self):
        """
        Main loop for the background service.
        This method must be implemented by subclasses.
        
        This should be a loop that checks self._stop_event regularly
        and exits when it's set.
        """
        pass
    
    def start(self):
        """Start the background service thread"""
        if self._thread is not None and self._thread.is_alive():
            self.logger.warning(f"Service {self.name} already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_wrapper, daemon=True)
        self._thread.start()
        self.logger.info(f"Started background service: {self.name}")
    
    def stop(self):
        """Stop the background service thread"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.logger.info(f"Stopped background service: {self.name}")
    
    def _run_wrapper(self):
        """Internal wrapper with error handling"""
        try:
            self.run()
        except Exception as e:
            self.logger.error(f"Error in background service {self.name}: {e}")


class EventModule(BaseChatRFModule):
    """
    Base class for event-driven modules.
    
    These modules respond to specific events in the repeater lifecycle.
    
    Events:
        - on_transmission_start: When someone starts talking
        - on_transmission_end: When transmission ends (before roger beep)
        - on_carrier_detect: When carrier is first detected
        - on_silence: When silence is detected
    """
    
    description = "Event-driven module"
    
    def on_transmission_start(self):
        """Called when a transmission starts"""
        pass
    
    def on_transmission_end(self):
        """Called when a transmission ends"""
        pass
    
    def on_silence(self):
        """Called when silence is detected after talking"""
        pass


# Module registry - this will be populated by the module loader
REGISTERED_MODULES = {
    'dtmf': [],
    'periodic': [],
    'service': [],
    'event': []
}


def register_module(module_class):
    """
    Decorator to register a module.
    The loader will find modules automatically,
    but this provides another way to register.
    
    Usage:
        @register_module
        class MyModule(DTMFModule):
            ...
    """
    # Determine module type
    if issubclass(module_class, DTMFModule):
        REGISTERED_MODULES['dtmf'].append(module_class)
    elif issubclass(module_class, PeriodicModule):
        REGISTERED_MODULES['periodic'].append(module_class)
    elif issubclass(module_class, BackgroundServiceModule):
        REGISTERED_MODULES['service'].append(module_class)
    elif issubclass(module_class, EventModule):
        REGISTERED_MODULES['event'].append(module_class)
    
    return module_class