# TTS-script

Fast local neural text-to-speech using Piper TTS with automatic Hugging Face model downloading, chunking for long texts, and progress tracking.

---

## Installation & Setup

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/Spondle/TTS-script.git
   cd TTS-script
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Enable global `tts` terminal command (optional):**
   Run the installer once to link `tts` into your `~/.local/bin`:
   ```bash
   ./install.sh
   ```
   *(Ensure `~/.local/bin` is in your `$PATH` if it isn't already).*

---

## Run Anywhere via Terminal (`tts`)

Once installed, you can run `tts` from **any directory** without needing to activate the virtual environment manually:

```bash
# Speak text directly
tts -t "Hello world"

# Read any text file and specify output
tts -f my_document.txt -o audio.wav

# Use Ryan's voice with 15% faster speed
tts -f my_document.txt -m en_US-ryan-medium --speed 1.15
```

---

## Quick Start: Synthesizing `input.txt`

### Option 1: In PyCharm
1. Open [`input.txt`](input.txt) and paste your text.
2. Right-click [`main.py`](main.py) and click **Run 'main'** (or press `Shift + F10`).
3. The generated audio will be saved to [`output_piper.wav`](output_piper.wav).

---

### Option 2: In Terminal

From the project directory:

```bash
# Run directly (reads input.txt by default)
python main.py
```

The script will synthesize the contents of [`input.txt`](input.txt) and output [`output_piper.wav`](output_piper.wav).

---

## Usage Options & Examples

### 1. Read from a Different File or Change Output
```bash
tts -f chapter1.txt -o chapter1.wav
```

### 2. Pass Text Directly in the Command Line
```bash
tts -t "Hello! This is a test."
```

### 3. Change Voice Models (Auto-Downloaded from Hugging Face)
If you don't have the model locally, it will automatically download it on first run:
```bash
# Ryan (Fast, warm male voice - recommended for long audiobooks)
tts -f input.txt -m en_US-ryan-medium -o ryan.wav

# Amy (Clear, natural female voice)
tts -f input.txt -m en_US-amy-medium -o amy.wav

# Bryce (Dynamic conversational male voice)
tts -f input.txt -m en_US-bryce-medium -o bryce.wav
```

### 4. Adjust Speed, Sentence Pauses, and Parallel Workers
```bash
# 15% faster speech rate with 0.1s silence between sentences and 4 parallel worker threads
tts -f input.txt --speed 1.15 --pause 0.1 --workers 4

# Run with specific hardware execution provider (auto, cpu, cuda, openvino)
tts -f input.txt --device auto
```

---

## Command-Line Arguments Reference

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--file` | `-f` | `input.txt` | Path to the input text file (or `-` for stdin) |
| `--output` | `-o` | `output_piper.wav` | Output `.wav` audio destination |
| `--text` | `-t` | `None` | Text string to synthesize directly |
| `--model` | `-m` | `en_US-ryan-medium.onnx` | Voice model name or path |
| `--speed` | | `1.0` | Speed multiplier (e.g. `1.15` for 15% faster) |
| `--pause` | | `0.2` | Seconds of silence between sentences |
| `--workers` | `-w` | `min(4, CPU count)` | Parallel worker threads for synthesis |
| `--device` | | `auto` | ONNX Runtime provider (`auto`, `cpu`, `cuda`, `openvino`) |
| `--speaker` | | `None` | Speaker ID (for multi-speaker models) |

---

## License

This project is licensed under the [MIT License](LICENSE).
