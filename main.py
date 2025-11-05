import wave
import sys
from piper.voice import PiperVoice

MODEL_FILE = "en_US-lessac-high.onnx"
INPUT_TEXT_FILE = "input.txt"  # <-- Define the input file
OUTPUT_FILE = "output_piper.wav"

# --- NEW SECTION: Read text from file ---
try:
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        TEXT_TO_SPEAK = f.read()
    print(f"Loaded text from: {INPUT_TEXT_FILE}")
    
    # Optional: Clean up whitespace
    TEXT_TO_SPEAK = TEXT_TO_SPEAK.strip() 

except FileNotFoundError:
    print(f"Error: Input text file not found at '{INPUT_TEXT_FILE}'")
    sys.exit(1)
except Exception as e:
    print(f"Error reading text file: {e}")
    sys.exit(1)
# ----------------------------------------

try:
    voice = PiperVoice.load(MODEL_FILE)
    print(f"Loaded voice model: {MODEL_FILE}")
except Exception as e:
    print(f"Error loading voice model: {e}")
    sys.exit(1)

if voice.config is None:
    print(f"Error: Model config file (.json) not found for '{MODEL_FILE}'.")
    print(f"Make sure '{MODEL_FILE}.json' is in the same directory.")
    sys.exit(1)

print(f"Model config loaded. Sample rate: {voice.config.sample_rate}")

try:
    # 1. Get the audio generator
    audio_generator = voice.synthesize(TEXT_TO_SPEAK)

    # 2. Open the file
    with wave.open(OUTPUT_FILE, "wb") as wav_file:
        
        # 3. Set the parameters
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(voice.config.sample_rate)
        
        # 4. Loop through the generator and write the correct attribute
        print("Synthesizing and writing audio chunks...")
        for audio_chunk in audio_generator:
            wav_file.writeframes(audio_chunk.audio_int16_bytes)
        
        print("Finished writing chunks.")

    print(f"Audio saved to: {OUTPUT_FILE}")

except Exception as e:
    print(f"An error occurred: {e}")