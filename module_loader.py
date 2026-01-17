"""
ChatRF Module Loader System

This handles:
1. Auto-discovery and loading of modules
2. Dynamic flag registration for each module
3. Integration with the main repeater loop
"""

import importlib
import inspect
import logging
from pathlib import Path
from modules.base import (
    DTMFModule, 
    PeriodicModule, 
    BackgroundServiceModule, 
    EventModule
)

logger = logging.getLogger(__name__)


class ModuleManager:
    """
    Manages all ChatRF modules: loading, flags, and execution.
    """
    
    def __init__(self, repeater):
        self.repeater = repeater
        
        # Module storage
        self.dtmf_modules = {}        # {command: module_instance}
        self.periodic_modules = []    # [module_instance, ...]
        self.service_modules = []     # [module_instance, ...]
        self.event_modules = []       # [module_instance, ...]
        
        # Dynamic flags, each module gets its own flag
        self.module_flags = {}        # {flag_name: False}
        self.flag_to_module = {}      # {flag_name: module_instance}
        
    def load_all_modules(self, modules_dir="modules"):
        """
        Discover and load all modules from the modules directory.
        
        Args:
            modules_dir: Path to the modules directory
        """
        logger.info("--------------------------")
        logger.info(f"Loading custom modules...")
        
        modules_path = Path(modules_dir)
        if not modules_path.exists():
            logger.warning(f"Modules directory not found: {modules_dir}")
            return
        
        # Scan for Python files
        for module_file in modules_path.glob("*.py"):
            # Skip __init__.py and base.py
            if module_file.name.startswith("_") or module_file.name == "base.py":
                continue
            
            module_name = f"{modules_dir}.{module_file.stem}"
            self._load_module(module_name)
        
        # Log summary
        logger.info(f"Module loading complete:")
        logger.info(f"  DTMF modules: {len(self.dtmf_modules)}")
        logger.info(f"  Periodic modules: {len(self.periodic_modules)}")
        logger.info(f"  Service modules: {len(self.service_modules)}")
        logger.info(f"  Event modules: {len(self.event_modules)}")
    
    def _load_module(self, module_name):
        """Load a single module and register it"""
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip if it's imported from another module
                if obj.__module__ != module_name:
                    continue
                
                # Check what type of module it is and register it
                if issubclass(obj, DTMFModule) and obj != DTMFModule:
                    self._register_dtmf_module(obj)
                
                elif issubclass(obj, PeriodicModule) and obj != PeriodicModule:
                    self._register_periodic_module(obj)
                
                elif issubclass(obj, BackgroundServiceModule) and obj != BackgroundServiceModule:
                    self._register_service_module(obj)
                
                elif issubclass(obj, EventModule) and obj != EventModule:
                    self._register_event_module(obj)
        
        except Exception as e:
            logger.error(f"Failed to load module {module_name}: {e}", exc_info=True)
    
    def _register_dtmf_module(self, module_class):
        """Register a DTMF module"""
        try:
            instance = module_class(self.repeater)
            
            if not instance.enabled:
                logger.info(f"Skipping disabled module: {instance.name}")
                return
            
            command = instance.dtmf_command
            
            # Check for command conflicts
            if command in self.dtmf_modules:
                logger.warning(
                    f"DTMF command '{command}' conflict: "
                    f"{instance.name} vs {self.dtmf_modules[command].name}. "
                    f"Keeping {self.dtmf_modules[command].name}"
                )
                return
            
            # Check if command is disabled in config
            if command in self.repeater.config.DISABLED_DTMF_COMMANDS:
                logger.info(f"DTMF module {instance.name} disabled via config (command: {command})")
                return
            
            # Register the module
            self.dtmf_modules[command] = instance
            
            # Create a dynamic flag for this module
            flag_name = self._create_flag_name(instance.name)
            self.module_flags[flag_name] = False
            self.flag_to_module[flag_name] = instance
            
            # Initialize the module
            instance.initialize()
            
            logger.info(
                f"Loaded DTMF module: {instance.name} "
                f"(Command: '{command}')"
            )
            
        except Exception as e:
            logger.error(f"Error registering DTMF module {module_class}: {e}", exc_info=True)
    
    def _register_periodic_module(self, module_class):
        """Register a periodic module"""
        try:
            instance = module_class(self.repeater)
            
            if not instance.enabled:
                logger.info(f"Skipping disabled periodic module: {instance.name}")
                return
            
            self.periodic_modules.append(instance)
            
            # Initialize and start the module
            instance.initialize()
            instance.start()
            
            logger.info(
                f"Loaded periodic module: {instance.name} "
                f"(Interval: {instance.interval_seconds}s)"
            )
            
        except Exception as e:
            logger.error(f"Error registering periodic module {module_class}: {e}", exc_info=True)
    
    def _register_service_module(self, module_class):
        """Register a background service module"""
        try:
            instance = module_class(self.repeater)
            
            if not instance.enabled:
                logger.info(f"Skipping disabled service module: {instance.name}")
                return
            
            self.service_modules.append(instance)
            
            # Initialize and start the module
            instance.initialize()
            instance.start()
            
            logger.info(f"Loaded service module: {instance.name}")
            
        except Exception as e:
            logger.error(f"Error registering service module {module_class}: {e}", exc_info=True)
    
    def _register_event_module(self, module_class):
        """Register an event module"""
        try:
            instance = module_class(self.repeater)
            
            if not instance.enabled:
                logger.info(f"Skipping disabled event module: {instance.name}")
                return
            
            self.event_modules.append(instance)
            instance.initialize()
            
            logger.info(f"Loaded event module: {instance.name}")
            
        except Exception as e:
            logger.error(f"Error registering event module {module_class}: {e}", exc_info=True)
    
    def _create_flag_name(self, module_name):
        """
        Create a flag name from module name.
        Example: "Fun Fact" -> "play_fun_fact"
        """
        # Convert to lowercase, replace spaces with underscores
        flag_name = module_name.lower().replace(" ", "_").replace("-", "_")
        # Add play_ prefix
        flag_name = f"play_{flag_name}"
        return flag_name
    
    def handle_dtmf_command(self, command):
        """
        Handle a DTMF command by routing it to the appropriate module.
        
        Args:
            command: The DTMF command received (e.g., '4')
            
        Returns:
            bool: True if command was handled, False if not found
        """
        module = self.dtmf_modules.get(command)
        
        if not module:
            return False
        
        # Check if module can execute
        if not module.can_execute():
            logger.info(f"Module {module.name} cannot execute right now")
            return True  # Still handled, but blocked
        
        # Check rate limiting if required
        if module.requires_rate_limit:
            if not self.repeater.command_allowed():
                return True  # Rate limited, but command was recognized
        
        try:
            # Execute the module's command handler
            module.handle_command()
            return True
            
        except Exception as e:
            logger.error(f"Error executing DTMF module {module.name}: {e}", exc_info=True)
            return True
    
    def check_flags(self):
        """
        Check all module flags and execute any that are set.
        This is called by the main loop after transmission ends.
        
        Returns:
            bool: True if any flag was processed
        """
        processed = False
        
        for flag_name, is_set in list(self.module_flags.items()):
            if is_set:
                module = self.flag_to_module[flag_name]
                
                try:
                    logger.debug(f"Processing flag: {flag_name} for module {module.name}")
                    
                    # Call the module's execution method
                    module.execute()
                    
                    processed = True
                    
                except Exception as e:
                    logger.error(f"Error processing flag {flag_name}: {e}", exc_info=True)
                
                finally:
                    # Clear the flag
                    self.module_flags[flag_name] = False
        
        return processed
    
    def set_flag(self, flag_name, value=True):
        """Set a module flag"""
        if flag_name in self.module_flags:
            self.module_flags[flag_name] = value
        else:
            logger.warning(f"Unknown flag: {flag_name}")
    
    def get_flag(self, flag_name):
        """Get a module flag value"""
        return self.module_flags.get(flag_name, False)
    
    def trigger_event(self, event_name, *args, **kwargs):
        """
        Trigger an event for all event modules.
        
        Args:
            event_name: Name of the event method (e.g., 'on_transmission_end')
        """
        for module in self.event_modules:
            try:
                if hasattr(module, event_name):
                    method = getattr(module, event_name)
                    method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event module {module.name}.{event_name}: {e}", exc_info=True)
    
    def shutdown_all(self):
        """Shutdown all modules gracefully"""
        logger.info("---------------------")
        logger.info("Shutting down all modules...")
        
        # Stop periodic modules
        for module in self.periodic_modules:
            try:
                module.stop()
                module.cleanup()
            except Exception as e:
                logger.error(f"Error stopping periodic module {module.name}: {e}")
        
        # Stop service modules
        for module in self.service_modules:
            try:
                module.stop()
                module.cleanup()
            except Exception as e:
                logger.error(f"Error stopping service module {module.name}: {e}")
        
        # Cleanup DTMF modules
        for module in self.dtmf_modules.values():
            try:
                module.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up DTMF module {module.name}: {e}")
        
        # Cleanup event modules
        for module in self.event_modules:
            try:
                module.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up event module {module.name}: {e}")
        
        logger.info("All modules shut down")
        logger.info("---------------------")
    
    def get_dtmf_menu(self):
        """
        Generate a menu of DTMF commands for the menu audio.
        
        Returns:
            str: Human-readable menu text
        """
        menu_items = []
        
        # Sort by command number/symbol
        sorted_commands = sorted(self.dtmf_modules.items())
        
        for command, module in sorted_commands:
            menu_items.append(f"{command}: {module.description}")
        
        return ", ".join(menu_items)
