#!/usr/bin/env python3
import argparse
import os
import re
import sys
import urllib.request
import wave
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

# Automatically re-execute inside the project's .venv if run with system python
_venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
if _venv_python.exists() and sys.executable != str(_venv_python):
    try:
        import piper  # noqa: F401
    except ImportError:
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

from piper.config import SynthesisConfig
from piper.voice import PiperVoice

# HuggingFace repository base for Piper voices
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def get_huggingface_urls(model_name: str) -> tuple[str, str] | None:
    """Builds HuggingFace download URLs for .onnx and .onnx.json from a voice identifier."""
    base_name = model_name.removesuffix(".onnx").removesuffix(".json")
    parts = base_name.split("-")
    if len(parts) == 3:
        locale, voice_name, quality = parts
        lang = locale.split("_")[0]
        url_prefix = f"{HF_BASE_URL}/{lang}/{locale}/{voice_name}/{quality}/{base_name}"
        return f"{url_prefix}.onnx", f"{url_prefix}.onnx.json"
    return None


def download_with_progress(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL with a visual progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Piper-TTS-Script"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            with open(temp_path, "wb") as f, tqdm(
                desc=f"Downloading {dest_path.name}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            ) as bar:
                while chunk := response.read(64 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        temp_path.replace(dest_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def ensure_model_files(model_input: str, config_input: str | None = None) -> tuple[Path, Path]:
    """Ensures both the .onnx model and .onnx.json config exist locally, downloading if necessary.
    
    Checks current directory first, then the centralized script directory so models are shared.
    """
    raw_name = model_input if model_input.endswith(".onnx") else f"{model_input}.onnx"
    local_path = Path(raw_name)
    local_config = Path(config_input) if config_input else Path(f"{local_path}.json")

    # 1. If explicit relative/absolute path exists locally
    if local_path.exists() and local_config.exists():
        return local_path, local_config

    # 2. Check in centralized script directory so models can be reused from any terminal location
    script_model = SCRIPT_DIR / local_path.name
    script_config = Path(config_input) if config_input else SCRIPT_DIR / f"{script_model.name}.json"
    if script_model.exists() and script_config.exists():
        return script_model, script_config

    # 3. If neither exists, determine destination (centralized script directory if name only)
    dest_model = local_path if ("/" in model_input or "\\" in model_input) else script_model
    dest_config = local_config if ("/" in model_input or "\\" in model_input) else script_config

    urls = get_huggingface_urls(dest_model.name)
    if not urls:
        if not dest_model.exists():
            print(f"Error: Model file not found at '{dest_model}' and couldn't parse HuggingFace URL.", file=sys.stderr)
            sys.exit(1)
        if not dest_config.exists():
            print(f"Error: Config file not found at '{dest_config}'.", file=sys.stderr)
            sys.exit(1)
        return dest_model, dest_config

    onnx_url, json_url = urls

    # Download ONNX if missing
    if not dest_model.exists():
        print(f"Model '{dest_model.name}' not found locally. Downloading from Hugging Face...")
        try:
            download_with_progress(onnx_url, dest_model)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Download JSON config if missing
    if not dest_config.exists():
        print(f"Config '{dest_config.name}' not found locally. Downloading from Hugging Face...")
        try:
            download_with_progress(json_url, dest_config)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    return dest_model, dest_config


def split_text_into_chunks(text: str, max_chunk_chars: int = 1500) -> list[str]:
    """Splits long text into manageable paragraph/sentence chunks for streaming synthesis."""
    # First, split by paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= max_chunk_chars:
            chunks.append(para)
        else:
            # Sub-split paragraph by sentences if too large
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk = []
            current_len = 0
            for sentence in sentences:
                if current_len + len(sentence) > max_chunk_chars and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_len = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_len += len(sentence) + 1
            if current_chunk:
                chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def synthesize_speech(
    text: str,
    output_path: str = "output_piper.wav",
    model_path: str = "en_US-lessac-high.onnx",
    config_path: str | None = None,
    speaker_id: int | None = None,
    length_scale: float = 1.0,      # Speech rate (<1.0 faster, >1.0 slower)
    noise_scale: float = 0.667,     # Phoneme variation
    noise_w: float = 0.8,           # Cadence variation
    sentence_silence: float = 0.2,  # Pause between sentences (seconds)
) -> None:
    """Synthesizes text (including very long texts) to a WAV file using Piper TTS."""
    resolved_model, resolved_config = ensure_model_files(model_path, config_path)

    # Load Piper voice model
    try:
        voice = PiperVoice.load(str(resolved_model), config_path=str(resolved_config))
    except Exception as e:
        print(f"Error loading voice model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded voice model: {resolved_model.name} (Sample rate: {voice.config.sample_rate} Hz)")

    # Configure synthesis parameters
    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w,
    )

    # Split text into chunks for streaming synthesis and progress reporting
    chunks = split_text_into_chunks(text)
    total_chars = sum(len(c) for c in chunks)
    print(f"Total characters: {total_chars:,} across {len(chunks)} chunks.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(voice.config.sample_rate)

            with tqdm(
                total=len(chunks),
                desc="Synthesizing audio",
                unit="chunk",
                dynamic_ncols=True,
            ) as progress:
                for chunk in chunks:
                    for audio_chunk in voice.synthesize(chunk, syn_config=syn_config):
                        wav_file.writeframes(audio_chunk.audio_int16_bytes)
                    progress.update(1)

        print(f"Done! Audio successfully saved to: {output_file.resolve()}")
    except Exception as e:
        print(f"Error during synthesis: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Piper TTS Generator - Fast local neural text-to-speech for short and long texts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--text", help="Text string to speak directly")
    parser.add_argument("-f", "--file", help="Path to input text file (or '-' for stdin)", default=None)
    parser.add_argument("-o", "--output", help="Output WAV path", default="output_piper.wav")
    parser.add_argument(
        "-m",
        "--model",
        help="Voice model name or .onnx path (e.g., en_US-lessac-high, en_US-ryan-medium, en_US-amy-medium)",
        default="en_US-lessac-high.onnx",
    )
    parser.add_argument("-c", "--config", help="Path to model config .json (optional)", default=None)
    parser.add_argument("--speaker", type=int, default=None, help="Speaker ID (for multi-speaker models)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (e.g. 1.2 for faster)")
    parser.add_argument("--pause", type=float, default=0.2, help="Silence between sentences in seconds")

    args = parser.parse_args()

    # Determine input text source
    text = ""
    if args.text:
        text = args.text
    elif args.file == "-":
        text = sys.stdin.read()
    else:
        # If user explicitly passed -f <file>
        if args.file:
            input_path = Path(args.file)
            if not input_path.exists():
                print(f"Error: Input text file '{input_path}' not found.", file=sys.stderr)
                sys.exit(1)
            text = input_path.read_text(encoding="utf-8")
        else:
            # Default to input.txt in current working dir or project dir if present
            default_file = Path("input.txt")
            if not default_file.exists() and (SCRIPT_DIR / "input.txt").exists():
                default_file = SCRIPT_DIR / "input.txt"
            if default_file.exists():
                text = default_file.read_text(encoding="utf-8")
            else:
                print("Error: No text input provided.", file=sys.stderr)
                print("Provide text via: tts -t \"your text\", or tts -f <filename>, or create input.txt.", file=sys.stderr)
                sys.exit(1)

    text = text.strip()
    if not text:
        print("Error: No text provided to synthesize.", file=sys.stderr)
        sys.exit(1)

    # Invert speed multiplier for Piper's length_scale parameter
    length_scale = (1.0 / args.speed) if args.speed > 0 else 1.0

    synthesize_speech(
        text=text,
        output_path=args.output,
        model_path=args.model,
        config_path=args.config,
        speaker_id=args.speaker,
        length_scale=length_scale,
        sentence_silence=args.pause,
    )


if __name__ == "__main__":
    main()
