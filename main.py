#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import sys
import urllib.request
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Automatically re-execute inside the project's .venv if run with system python
_venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
if _venv_python.exists() and sys.executable != str(_venv_python):
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import piper  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError:
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

import numpy as np
import onnxruntime as ort
from piper.config import PiperConfig, SynthesisConfig
from piper.voice import PiperVoice
from tqdm import tqdm

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


def resolve_providers(device: str) -> list[str | tuple[str, dict]]:
    """Resolves ONNX Runtime execution providers based on requested device."""
    available = set(ort.get_available_providers())
    dev = device.lower()

    if dev == "cuda" or (dev == "auto" and "CUDAExecutionProvider" in available):
        if "CUDAExecutionProvider" not in available:
            print("Warning: CUDAExecutionProvider not available, falling back to CPU.", file=sys.stderr)
        else:
            return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]

    if dev == "openvino" or (dev == "auto" and "OpenVINOExecutionProvider" in available):
        if "OpenVINOExecutionProvider" not in available:
            print("Warning: OpenVINOExecutionProvider not available, falling back to CPU.", file=sys.stderr)
        else:
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def load_optimized_voice(
    model_path: Path,
    config: PiperConfig,
    device: str = "auto",
    workers: int = 1,
) -> PiperVoice:
    """Loads a single thread-safe PiperVoice session with ORT graph optimizations."""
    cpu_count = os.cpu_count() or 4
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.intra_op_num_threads = max(1, cpu_count // max(1, workers))
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True

    providers = resolve_providers(device)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )
    return PiperVoice(
        config=config,
        session=session,
    )


def extract_sentence_phoneme_ids(config: PiperConfig, text: str) -> list[list[int]]:
    """Splits text by paragraphs and uses Espeak to extract per-sentence phoneme IDs."""
    # Phonemization only requires PiperConfig (does not use the ONNX session)
    dummy_voice = PiperVoice(config=config, session=None)  # type: ignore[arg-type]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    sentence_batches: list[list[int]] = []
    for para in paragraphs:
        for phonemes in dummy_voice.phonemize(para):
            if phonemes:
                sentence_batches.append(dummy_voice.phonemes_to_ids(phonemes))
    return sentence_batches


def _synthesize_sentence_pcm(
    voice: PiperVoice,
    phoneme_ids: list[int],
    syn_config: SynthesisConfig,
) -> bytes:
    """Runs thread-safe ORT inference and fused float32 -> int16 PCM conversion."""
    audio = voice.phoneme_ids_to_audio(phoneme_ids, syn_config=syn_config)
    if isinstance(audio, tuple):
        audio = audio[0]

    if audio.size == 0:
        return b""

    if syn_config.normalize_audio:
        max_val = float(np.max(np.abs(audio)))
        if max_val < 1e-8:
            return bytes(audio.size * 2)
        scale = (32767.0 * syn_config.volume) / max_val
    else:
        scale = 32767.0 * syn_config.volume

    return np.clip(audio * scale, -32767.0, 32767.0).astype(np.int16).tobytes()


def synthesize_speech(
    text: str,
    output_path: str = "output_piper.wav",
    model_path: str = "en_US-ryan-medium.onnx",
    config_path: str | None = None,
    speaker_id: int | None = None,
    length_scale: float = 1.0,      # Speech rate (<1.0 faster, >1.0 slower)
    noise_scale: float = 0.667,     # Phoneme variation
    noise_w: float = 0.8,           # Cadence variation
    sentence_silence: float = 0.2,  # Pause between sentences (seconds)
    workers: int | None = None,     # Concurrency count
    device: str = "auto",           # Hardware execution provider
) -> None:
    """Synthesizes text to a WAV file using parallel Piper TTS with thread-safe inference."""
    resolved_model, resolved_config = ensure_model_files(model_path, config_path)

    cpu_count = os.cpu_count() or 4
    if workers is None or workers <= 0:
        workers = min(4, cpu_count)

    # Load voice config and extract sentence phoneme batches first so thread count
    # scales accurately when synthesizing only 1-3 sentences
    try:
        with open(resolved_config, "r", encoding="utf-8") as f:
            voice_config = PiperConfig.from_dict(json.load(f))
        sentence_batches = extract_sentence_phoneme_ids(voice_config, text)
    except Exception as e:
        print(f"Error preparing voice config or phonemizing text: {e}", file=sys.stderr)
        sys.exit(1)

    total_sentences = len(sentence_batches)
    if total_sentences == 0:
        print("Error: No synthesizeable sentences found in input text.", file=sys.stderr)
        sys.exit(1)

    effective_workers = min(workers, total_sentences)

    # Load thread-safe Piper voice model with intra_op_num_threads matched to effective_workers
    try:
        voice = load_optimized_voice(
            model_path=resolved_model,
            config=voice_config,
            device=device,
            workers=effective_workers,
        )
    except Exception as e:
        print(f"Error loading voice model: {e}", file=sys.stderr)
        sys.exit(1)

    active_provider = voice.session.get_providers()[0]
    print(
        f"Loaded voice model: {resolved_model.name} "
        f"(Sample rate: {voice.config.sample_rate} Hz, Provider: {active_provider})"
    )

    total_phonemes = sum(len(b) for b in sentence_batches)
    print(
        f"Prepared {total_sentences} sentences ({total_phonemes:,} phoneme IDs). "
        f"Synthesizing with {effective_workers} worker thread(s)..."
    )

    # Configure synthesis parameters
    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w,
    )

    # Silence padding between sentences
    silence_samples = max(0, int(voice.config.sample_rate * sentence_silence))
    silence_bytes = b"\x00\x00" * silence_samples if silence_samples > 0 else b""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Windowed Longest-Job-First (LJF) scheduling: prevents tail stragglers within each
    # window while bounding in-memory out-of-order audio buffering for huge texts
    window_size = max(16, effective_workers * 8)
    indexed_batches = list(enumerate(sentence_batches))
    scheduled_jobs: list[tuple[int, list[int]]] = []
    for start in range(0, total_sentences, window_size):
        window = indexed_batches[start : start + window_size]
        window.sort(key=lambda item: len(item[1]), reverse=True)
        scheduled_jobs.extend(window)

    completed: dict[int, bytes] = {}
    next_write_idx = 0

    try:
        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(voice.config.sample_rate)

            with tqdm(
                total=total_sentences,
                desc="Synthesizing audio",
                unit="sentence",
                dynamic_ncols=True,
            ) as progress:
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = {
                        executor.submit(
                            _synthesize_sentence_pcm, voice, pids, syn_config
                        ): idx
                        for idx, pids in scheduled_jobs
                    }
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        completed[idx] = future.result()
                        progress.update(1)

                        # Write buffered chunks in strict sequential order
                        while next_write_idx in completed:
                            wav_file.writeframes(completed.pop(next_write_idx))
                            if silence_bytes and next_write_idx < total_sentences - 1:
                                wav_file.writeframes(silence_bytes)
                            next_write_idx += 1

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
        help="Voice model name or .onnx path (e.g., en_US-ryan-medium, en_US-lessac-high, en_US-amy-medium)",
        default="en_US-ryan-medium.onnx",
    )
    parser.add_argument("-c", "--config", help="Path to model config .json (optional)", default=None)
    parser.add_argument("--speaker", type=int, default=None, help="Speaker ID (for multi-speaker models)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (e.g. 1.2 for faster)")
    parser.add_argument("--pause", type=float, default=0.2, help="Silence between sentences in seconds")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of parallel synthesis worker threads",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "openvino"],
        default="auto",
        help="ONNX Runtime execution provider device",
    )

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
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
