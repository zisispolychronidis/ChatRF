<p align="center">
  <img src="https://i.imgur.com/thoXDHu.png" alt="ChatRF – AI Ham Radio Repeater Logo" width="80%"/>
</p>

# ChatRF – AI-Enhanced Ham Radio Repeater

**ChatRF** is a modular Python-based ham radio repeater system enhanced with an embedded conversational AI assistant. Designed for amateur radio enthusiasts, this project integrates real-time AI interaction, audio signal processing, DTMF command handling, weather updates, satellite tracking, callsign lookups, and more — all through standard RF equipment and a local computer (e.g., Raspberry Pi). It runs fully offline with a local LLM.

---

## 📡 What is ChatRF?

ChatRF is a hybrid hardware-software repeater controller that:
- Functions as a traditional RF repeater.
- Transmits automatic CW ID.
- Responds to DTMF touch-tone commands for live interaction.
- Offers conversational responses using a **local** LLM (default: `gemma3` via Ollama).
- Serves as a flexible framework for ham-related features accessible by voice.

---

## 🎯 Core Features

### ✅ Repeater System
- Detects and forwards incoming transmissions with optional silence detection.
- Periodic Morse code ID.
- Interacts via DTMF for feature control.

### 🤖 AI Assistant ("AI Mode")
- Records and transcribes voice using `faster-whisper`.
- Generates replies with a local LLM using Ollama.
- Speaks responses via `espeak-ng`.
- Plays a calming “thinking melody” during processing.
- Automatically exits on silence or timeout.
- Custom system prompt optimized for ham radio conversations.

### 🌐 Modules
- **Callsign Lookup**: Translates phonetic input into callsigns, then checks against the RadioID database.
- **Satellite Tracking**: Uses Skyfield + TLEs to report upcoming satellite passes.
- **HF Band Conditions**: Pulls solar/propagation info from hamqsl.com.
- **Weather Reports**: Via OpenWeatherMap API.
- **Voice Output**: All responses spoken using `espeak-ng`.

---

## 🔢 DTMF Command Menu

| DTMF Tone | Action                          |
|----------:|---------------------------------|
| `*`       | Toggle AI Assistant             |
| `#`       | Playback help menu              |
| `0`       | System information              |
| `1`       | Current time & date             |
| `2`       | Local weather forecast          |
| `3`       | HF band conditions              |
| `4`       | Random fun fact                 |
| `5`       | Callsign lookup (speech input)  |
| `6`       | Satellite pass predictions      |
| `7`       | Random meme sound 😂             |

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

All setup instructions, wiring diagrams, and deployment guidance are provided in the project's **GitHub Wiki**.

---

## 🤝 Contributions

Pull requests are welcome! If you'd like to contribute a module or bugfix, please open an issue or submit a PR.

---

### 📡 Developed by SV2TMT — Zisis Polychronidis

This is part of an ongoing exploration of merging cutting-edge AI with classic ham radio principles.  
**Made by young hams, for the future of amateur radio.**

---
