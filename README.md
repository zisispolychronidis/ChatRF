<p align="center">
  <img src="https://i.imgur.com/thoXDHu.png" alt="ChatRF Logo" width="80%"/>
</p>

# ChatRF – AI-Enhanced Ham Radio Repeater

**ChatRF** is a modular Python-based ham radio repeater system enhanced with an embedded conversational AI assistant. Designed for amateur radio enthusiasts, this project integrates real-time AI interaction, audio signal processing, DTMF command handling, weather info, satellite pass predictions, callsign lookups, and more—all through standard radio equipment and a computer (Raspberry Pi is supported).

## 📡 What is ChatRF?

ChatRF is a hybrid hardware-software project for a ham radio repeater that can:
- Act as a regular RF repeater system.
- Transmit station identification (CW ID) automatically.
- Accept DTMF commands for running different tasks.
- Answer amateur radio and general knowledge questions using a local LLM.
- Provide a framework for different applications, accessible through an amateur radio repeater system.

## 🎯 Core Features

### ✅ Repeater System
- Detects audio presence and forwards transmission with configurable silence detection.
- Periodically transmits callsign in Morse code.
- Responds to DTMF touch-tone commands for dynamic control.

### 🤖 AI Assistant ("AI Mode")
- Records and transcribes user speech using `faster-whisper`.
- Generates responses with Ollama using a local LLM (default: `gemma3`).
- Speaks responses back using `espeak-ng`.
- Plays a gentle “thinking melody” while processing.
- Automatically cancels after inactivity or silence.
- Uses a custom system prompt tailored for ham radio topics.

### 🌐 Additional Modules
- **Callsign Lookup**: Extracts and decodes phonetic/numeric speech into callsigns and checks against RadioID database.
- **Satellite Tracking**: Uses Skyfield and real-time TLEs to predict upcoming AOS events.
- **Band Condition Reports**: Fetches current solar and band propagation data from hamqsl.com.
- **Weather**: Retrieves weather reports using the OpenWeatherMap API.
- **Voice Output**: Uses `espeak-ng` for all text-to-speech tasks.

### 🔢 Default DTMF Command Menu
| DTMF Tone | Action                         |
|----------:|--------------------------------|
| `*`       | Toggle AI Assistant            |
| `#`       | Play menu overview             |
| `0`       | Repeater information           |
| `1`       | Current time & date            |
| `2`       | Local weather forecast         |
| `3`       | HF band conditions             |
| `4`       | Random fun fact                |
| `5`       | Callsign lookup via speech     |
| `6`       | Next satellite pass predictions |
| `7`       | Play a random meme audio 😂       |

## 🔧 Modular Design

ChatRF is structured in a mostly modular way, allowing for:
- Easy extension or customization of new features.
- Adding new DTMF actions without modifying the core repeater loop heavily.
- Independent development of AI, repeater logic, audio playback, and other services.

## 🇬🇷 Language Support

- Originally designed for Greek-speaking amateur radio operators.
- Can be easily changed to use almost any language.
- Prioritizes clear, concise phrasing optimized for voice transmission over RF.

## 📖 Installation & Setup

Installation instructions, hardware requirements, audio wiring, and deployment tips will be available in the project's **GitHub Wiki**.

---

### 📡 Developed by SV2TMT — Zisis Polychronidis  
This project is part of an ongoing effort to explore the fusion of modern AI with traditional ham radio technology. Made by young people for the future generation of ham radio operators!

---

Feel free to suggest improvements, contribute, or fork it to create your own AI repeater system!
