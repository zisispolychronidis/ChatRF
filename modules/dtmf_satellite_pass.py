"""
Satellite Pass Module for ChatRF

This module announces upcoming satellite passes when DTMF command '6' is pressed.
"""

from datetime import datetime, timedelta
import pytz
import re
from pytz import timezone, utc
from skyfield.api import EarthSatellite, Topos, load, Loader
from modules.base import DTMFModule


class SatellitePassModule(DTMFModule):
    """
    Announces upcoming satellite passes for configured satellites.
    
    Uses Skyfield to calculate satellite positions and pass times.
    """
    
    # Module metadata
    name = "Satellite Pass"
    version = "1.0.5"
    description = "Satellite passes"
    
    # DTMF configuration
    dtmf_command = '6'
    requires_rate_limit = True
    
    def handle_command(self):
        """Execute when DTMF '6' is received."""

        # Schedule the satellite pass announcement
        flag_name = self.repeater.module_manager._create_flag_name(self.name)
        self.repeater.module_manager.set_flag(flag_name)
    
    def execute(self):
        """Standardized execution method called by ModuleManager."""

        self.play_satellite_pass()
    
    def play_satellite_pass(self):
        """Calculate and announce satellite passes"""
        self.repeater.speak_with_piper("Αναζήτηση δορυφορικών διελεύσεων.")

        try:
            # Observer position
            observer = Topos(
                latitude_degrees=self.config.LATITUDE, 
                longitude_degrees=self.config.LONGITUDE, 
                elevation_m=self.config.ELEVATION
            )
            local_tz = pytz.timezone(self.config.TIMEZONE)
            ts = load.timescale()
            now = ts.now()
            t_end = ts.utc((datetime.now(utc) + timedelta(hours=24)))

            # Fetch TLEs
            tle_text = self.repeater.fetch_tles()
            lines = tle_text.strip().splitlines()

            # Parse TLEs into Skyfield satellites
            satellites = []
            
            # Name matching function
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
                                    self.logger.debug(f"Added satellite: {name}")
                                except Exception as e:
                                    self.logger.error(f"Error creating satellite {name}: {e}")
                            i += 3  # Skip the 3 lines we just processed
                        else:
                            i += 1  # Just skip this line if it's not followed by TLE data
                    else:
                        i += 1
                else:
                    i += 1

            self.logger.info(f"Total satellites loaded: {len(satellites)}")

            loader = Loader("data/cache")  # Initialize custom path
            eph = loader("de421.bsp")  # Required by skyfield
            
            # Find upcoming AOS events for each satellite
            passes = []
            
            for sat in satellites:
                try:
                    times, events = sat.find_events(observer, now, t_end, altitude_degrees=10.0)
                    self.logger.debug(f"Found {len(times)} events for {sat.name}")
                    
                    for t, event in zip(times, events):
                        if event == 0:  # AOS
                            dt_utc = t.utc_datetime()
                            dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
                            passes.append((sat.name, dt_local))
                            self.logger.debug(f"AOS for {sat.name}: {dt_local}")
                            break  # Only take first AOS per satellite
                            
                except Exception as e:
                    self.logger.error(f"Error finding events for {sat.name}: {e}")

            if not passes:
                spoken = "Δεν βρέθηκαν επικείμενες διελεύσεις δορυφόρων."
            else:
                # Sort and limit
                passes.sort(key=lambda x: x[1])
                passes = list(dict.fromkeys(passes))
                        
                spoken_parts = []

                for name, dt in passes[:3]:
                    clean_name = re.sub(r'\([^)]*\)', '', name)
                    hour = dt.hour % 12
                    minute = dt.minute
                    hour_str = self.config.GREEK_HOUR_NAMES.get(hour, str(hour))
                    minute_str = f"{minute}" if minute != 0 else "ακριβώς"
                    if minute != 0:
                        time_phrase = f"{hour_str} και {minute_str}"
                    else:
                        time_phrase = f"{hour_str} ακριβώς"
                    
                    spoken_parts.append(f"{clean_name}: {time_phrase}")

                spoken = "Οι επόμενες διελεύσεις είναι: " + ", ".join(spoken_parts) + " ώρα Ελλάδος."

            self.logger.info(f"Speaking satellite passes: {spoken}")
            self.repeater.speak_with_piper(spoken)

        except Exception as e:
            self.logger.error(f"Error during satellite pass playback: {e}")
    
    def can_execute(self):
        """Check if the command can execute."""
        # Only execute if not in AI mode
        if self.repeater.ai_mode_running:
            self.logger.info("Satellite pass blocked - AI mode is active")
            return False
        
        return True
    
    def cleanup(self):
        """Called when module is unloaded or system shuts down"""
        self.logger.info("Satellite pass module shutting down")
