from modules.base import DTMFModule
import os
import time


class WeatherModule(DTMFModule):
    name = "Weather"
    version = "1.0.0"
    description = "Fetches and speaks current weather conditions"
    enabled = True

    dtmf_command = '2'
    requires_rate_limit = True

    def handle_command(self):
        flag_name = self.repeater.module_manager._create_flag_name(self.name)
        self.repeater.module_manager.set_flag(flag_name)

    def execute(self):
        try:
            import requests, json

            city = self.config.config.get('Weather', 'city', fallback='Serres,GR')
            api_key = self.config.config.get('Weather', 'api_key', fallback='your_api_key_here')
            cache_file = self.config.config.get('Weather', 'cache_file', fallback='data/cache/weather_cache.json')
            cache_duration = self.config.config.getint('Weather', 'cache_duration', fallback=900)

            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=el"

            cache_valid = False
            forecast = ""

            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    timestamp = cache_data.get("timestamp", 0)
                    if time.time() - timestamp < cache_duration:
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
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump({"forecast": forecast, "timestamp": time.time()}, f)
                else:
                    self.logger.warning(f"Failed to fetch weather: {response.status_code}")
                    forecast = "Δεν μπόρεσα να ανακτήσω τα δεδομένα καιρού."

            self.logger.info(f"Speaking weather: {forecast}")
            self.repeater.speak_with_piper(forecast)

        except Exception as e:
            self.logger.error(f"Error during weather playback: {e}")

    def can_execute(self):
        if self.repeater.ai_mode_running:
            return False
        return True
