from modules.base import DTMFModule
from datetime import datetime


class BandConditionsModule(DTMFModule):
    name = "Band Conditions"
    version = "1.0.0"
    description = "Fetches and speaks current HF/VHF band conditions"
    enabled = True

    dtmf_command = '3'
    requires_rate_limit = True

    def handle_command(self):
        flag_name = self.repeater.module_manager._create_flag_name(self.name)
        self.repeater.module_manager.set_flag(flag_name)

    def execute(self):
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

                current_hour = datetime.now().hour
                is_night = current_hour >= 21 or current_hour < 6
                time_label = "night" if is_night else "day"

                # HF conditions
                bands = data.find("calculatedconditions")
                band_reports = {}
                for band in bands.findall("band"):
                    if band.attrib.get("time") == time_label:
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

                band_phrases = []
                for key, condition in band_reports.items():
                    greek_band = greek_band_names.get(key, key)
                    greek_condition = greek_condition_names.get(condition, condition)
                    band_phrases.append(f"{greek_band}: {greek_condition}")
                band_phrase = ". ".join(band_phrases)

                time_phrase = "κατά τη διάρκεια της νύχτας" if is_night else "κατά τη διάρκεια της ημέρας"

                # VHF phenomena
                vhf_section = data.find("calculatedvhfconditions")
                vhf_phrases = []
                if vhf_section is not None:
                    greek_phenomenon_names = {
                        "vhf-aurora": "VHF Βόρειο Σέλας",
                        "E-Skip": "Σποραδική Έψιλον",
                    }
                    greek_location_names = {
                        "northern_hemi": "στο βόρειο ημισφαίριο",
                        "europe": "στην Ευρώπη στα 2 μέτρα",
                        "north_america": "στην Αμερική στα 2 μέτρα",
                        "europe_6m": "στην Ευρώπη στα 6 μέτρα",
                        "europe_4m": "στην Ευρώπη στα 4 μέτρα",
                    }
                    for phenomenon in vhf_section.findall("phenomenon"):
                        status = (phenomenon.text or "").strip()
                        if status.lower() == "band closed":
                            continue  # skip inactive phenomena
                        else:
                            status = "Ανοιχτή μπάντα"

                        name = phenomenon.attrib.get("name", "")
                        location = phenomenon.attrib.get("location", "")
                        greek_name = greek_phenomenon_names.get(name, name)
                        greek_location = greek_location_names.get(location, location)
                        vhf_phrases.append(f"{greek_name} {greek_location}: {status}")

                # Build the full spoken report
                full_report = (
                    f"Ο δείκτης ηλιακής ροής είναι {sfi}, "
                    f"ο δείκτης Κ είναι {kindex}, "
                    f"ο αριθμός των ηλιακών κηλίδων είναι {sunspots}, "
                    f"και η ακτινοβολία X-ray είναι {xray}. "
                    f"Θόρυβος σήματος: {noise}. "
                    f"Καταστάσεις HF {time_phrase}: {band_phrase}."
                )

                if vhf_phrases:
                    vhf_report = "Φαινόμενα VHF: " + ". ".join(vhf_phrases) + "."
                    full_report += " " + vhf_report

                self.logger.info(f"Speaking band conditions: {full_report}")
                self.repeater.speak_with_piper(full_report)
            else:
                self.logger.warning("Failed to fetch band condition XML data")
        except Exception as e:
            self.logger.error(f"Error during band conditions playback: {e}")

    def can_execute(self):
        if self.repeater.ai_mode_running:
            return False
        return True
