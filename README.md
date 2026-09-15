# TTS-script

Fast local neural text-to-speech using Piper TTS with automatic Hugging Face model downloading, chunking for long texts, and progress tracking.

---

## Quick Start: Synthesizing `input.txt`

### Option 1: In PyCharm
1. Open [`input.txt`](input.txt) and paste your text.
2. Right-click [`main.py`](main.py) and click **Run 'main'** (or press `Shift + F10`).
3. The generated audio will be saved to [`output_piper.wav`](output_piper.wav).

---

### Option 2: In Terminal

Activate your virtual environment and run `main.py`:

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Run the script (reads input.txt by default)
python main.py
```

The script will synthesize the contents of [`input.txt`](input.txt) and output [`output_piper.wav`](output_piper.wav).

---

## Usage Options & Examples

### 1. Read from a Different File or Change Output
```bash
python main.py -f chapter1.txt -o chapter1.wav
```

### 2. Pass Text Directly in the Command Line
```bash
python main.py -t "Hello! This is a test."
```

### 3. Change Voice Models (Auto-Downloaded from Hugging Face)
If you don't have the model locally, it will automatically download it on first run:
```bash
# Ryan (Fast, warm male voice - recommended for long audiobooks)
python main.py -f input.txt -m en_US-ryan-medium -o ryan.wav

# Amy (Clear, natural female voice)
python main.py -f input.txt -m en_US-amy-medium -o amy.wav

# Bryce (Dynamic conversational male voice)
python main.py -f input.txt -m en_US-bryce-medium -o bryce.wav
```

### 4. Adjust Speed and Sentence Pauses
```bash
# 15% faster speech rate with 0.1s silence between sentences
python main.py -f input.txt --speed 1.15 --pause 0.1
```

---

## Command-Line Arguments Reference

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--file` | `-f` | `input.txt` | Path to the input text file (or `-` for stdin) |
| `--output` | `-o` | `output_piper.wav` | Output `.wav` audio destination |
| `--text` | `-t` | `None` | Text string to synthesize directly |
| `--model` | `-m` | `en_US-lessac-high.onnx` | Voice model name or path |
| `--speed` | | `1.0` | Speed multiplier (e.g. `1.15` for 15% faster) |
| `--pause` | | `0.2` | Seconds of silence between sentences |
| `--speaker` | | `None` | Speaker ID (for multi-speaker models) |
