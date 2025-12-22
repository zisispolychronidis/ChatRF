<p align="center">
  <img src="https://i.imgur.com/thoXDHu.png" alt="ChatRF – AI Ham Radio Repeater Logo" width="80%"/>
</p>

# ChatRF: AI-Enhanced Ham Radio Repeater

**ChatRF** is a modular Python-based ham radio repeater system enhanced with an embedded conversational AI assistant.
Designed for amateur radio enthusiasts, this project integrates real-time AI interaction, audio signal processing, DTMF command handling, weather updates, satellite tracking, callsign lookups, and more, all through standard RF equipment and a local computer (e.g., Raspberry Pi).

### 📚 Table of Contents

- [What is ChatRF?](#-what-is-chatrf)
- [Core Features](#-core-features)
  - [Repeater System](#-repeater-system)
  - [AI Assistant ("AI Mode")](#-ai-assistant-ai-mode)
- [DTMF Command Menu](#-dtmf-command-menu)
- [Installation & Setup](#-installation--setup)
- [Contributions](#-contributions)
- [Developed by](#-developed-by-sv2tmt--zisis-polychronidis)

---

## 📡 What is ChatRF?

ChatRF is a software controller for repeater systems that:
- Allows traditional RF repeater functionality.
- Transmits automatic CW ID.
- Responds to DTMF touch-tone commands for live interaction.
- Offers conversational responses using a **local** LLM (default: `gemma3` via Ollama).
- Serves as a flexible framework for ham-related features accessible by voice.

### ❓ Why ChatRF?

Ham radio has always been about innovation, but repeaters have barely changed since the 90s. ChatRF brings smart interaction to the airwaves, letting hams access real-time information, converse with an LLM, and explore modular features entirely over RF, without the internet.

---

## 🎯 Core Features

### ✅ Repeater System
- Detects and forwards incoming transmissions with changeable silence detection.
- Periodic Morse code ID.
- Interacts via DTMF for feature control.

### 🤖 AI Assistant ("AI Mode")
- Records and transcribes voice using `faster-whisper`.
- Generates replies with a local LLM using Ollama.
- Runs entirely offline — no internet required once set up.
- Speaks responses via `piper-tts`.
- Plays a “thinking melody” during processing.
- Automatically exits on timeout.
- Custom system prompt optimized for ham radio conversations.

### 🌐 Other Modules
- **Callsign Lookup**: Translates phonetic alphabet input into callsigns, then checks against the RadioID database.
- **Satellite Tracking**: Uses Skyfield + TLEs to report upcoming satellite passes.
- **HF Band Conditions**: Pulls solar/propagation info from hamqsl.com.
- **Weather Reports**: Via OpenWeatherMap API.
- **Voice Output**: All responses spoken using `piper-tts`.

### 🪄 Launch Arguments
- `--no-audio-repeat`: Disable audio repeating (monitor mode only)
- `--no-roger`: Disable roger beep at end of transmission
- `--no-cw-id`: Disable periodic CW callsign identification
- `--no-voice-id`: Disable periodic voice/audio station identification
- `--disable-dtmf [CMD ...]`: Disable specific DTMF commands
- `--no-dtmf`: Disable DTMF command listening
- `--no-ptt`: Disable PTT control via Raspberry Pi GPIO **(REQUIRED on Windows)**
- `--audio-boost AUDIO_BOOST`: Audio boost factor (default: 5.0)
- `--threshold THRESHOLD`: Audio detection threshold (default: 500)

---

## 🔢 DTMF Command Menu

| DTMF Tone | Action                          |
|----------:|---------------------------------|
| `*`       | Toggle AI Assistant             |
| `#`       | Playback help menu              |
| `0`       | Repeater information            |
| `1`       | Current time & date             |
| `2`       | Local weather forecast          |
| `3`       | HF band conditions              |
| `4`       | Random fun fact                 |
| `5`       | Callsign lookup (speech input)  |
| `6`       | Satellite pass predictions      |
| `7`       | Random meme sound 😂            |

---

## 🛠 Modular Design

- Easy to extend — plug in your own DTMF features.
- Independent modules for audio, logic, AI, etc.
- Designed with maintainability and experimentation in mind.

---

## 🌍 Language Support

- Default language: Greek.
- Easy to customize for any spoken language.
- AI responses are phrased for maximum intelligibility over radio.

---

## 📦 Installation & Setup

All setup instructions, development guidance, wiring diagrams, and deployment guidance are going to be provided in the project's **[GitHub Wiki](https://github.com/zisispolychronidis/ChatRF/wiki)**.

---

## 🤝 Contributions

Pull requests are welcome! If you'd like to contribute a module or bugfix, please open an issue or submit a PR.

Also, if you end up using this for your own ham radio station, I'd love to see it!

---

### 📡 Developed by SV2TMT, Zisis Polychronidis

This is part of an ongoing exploration of merging cutting-edge AI with classic ham radio principles.  
**Made by young hams, for the future of amateur radio.**

---
