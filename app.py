from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import ctranslate2
import streamlit as st
from faster_whisper import WhisperModel

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "aac"]
MODEL_OPTIONS = {
    "Large v3 Turbo (recommended for CPU)": "large-v3-turbo",
    "Large v3 (highest compute cost)": "large-v3",
    "Medium (faster)": "medium",
    "Small (fastest listed option)": "small",
}
LANGUAGES: dict[str, str | None] = {
    "English (default)": "en",
    "Auto-detect": None,
    "Arabic": "ar",
    "Chinese": "zh",
    "Dutch": "nl",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Marathi": "mr",
    "Portuguese": "pt",
    "Spanish": "es",
}
UK_MAP = {
    "color": "colour", "colors": "colours", "colored": "coloured",
    "favorite": "favourite", "favorites": "favourites",
    "behavior": "behaviour", "behaviors": "behaviours",
    "neighbor": "neighbour", "neighbors": "neighbours",
    "neighborhood": "neighbourhood", "honor": "honour",
    "defense": "defence", "offense": "offence",
    "analyze": "analyse", "analyzed": "analysed", "analyzing": "analysing",
    "apologize": "apologise", "realize": "realise", "realized": "realised",
    "traveler": "traveller", "travelers": "travellers",
    "traveling": "travelling", "traveled": "travelled",
    "canceled": "cancelled", "canceling": "cancelling",
    "gray": "grey", "airplane": "aeroplane", "aluminum": "aluminium",
    "jewelry": "jewellery",
}


def uk_english(text: str) -> str:
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, sorted(UK_MAP, key=len, reverse=True))) + r")\b",
        re.I,
    )

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        target = UK_MAP[source.lower()]
        if source.isupper():
            return target.upper()
        if source[:1].isupper():
            return target.capitalize()
        return target

    return pattern.sub(replace, text)


def output_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "audio"
    return f"{stem}_transcript_uk.txt"


def ffmpeg_status() -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return True, ffmpeg
    return False, ""


def audio_duration(path: str) -> float | None:
    if not shutil.which("ffprobe"):
        return None

    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
        return float(payload["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def convert_to_whisper_wav(source_path: str, wav_path: str) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", source_path, "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=None,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "Unknown FFmpeg decoding error").strip()
        raise RuntimeError(
            "FFmpeg could not decode this upload. The file may be incomplete, "
            f"corrupted, or contain no audio stream.\n\n{detail}"
        ) from exc

    output = Path(wav_path)
    if not output.exists() or output.stat().st_size <= 44:
        raise RuntimeError("FFmpeg did not produce a valid WAV file.")


def cuda_device_count() -> int:
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


@st.cache_resource(show_spinner=False)
def load_whisper(
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
) -> WhisperModel:
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def count_tokens(model: WhisperModel, text: str) -> int:
    if not text:
        return 0
    encoded: Any = model.hf_tokenizer.encode(text)
    return len(encoded.ids)


st.set_page_config(page_title="Local Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Local Whisper Transcriber")
st.caption(
    "Local transcription using faster-whisper/CTranslate2. "
    "No paid transcription API is used."
)

ffmpeg_ready, ffmpeg_path = ffmpeg_status()
cuda_count = cuda_device_count()
logical_cpus = os.cpu_count() or 4
default_threads = max(1, min(8, logical_cpus))

with st.sidebar:
    model_label = st.selectbox("Whisper model", list(MODEL_OPTIONS))
    language_label = st.selectbox("Spoken language", list(LANGUAGES))
    normalise = st.checkbox("Apply conservative UK spelling", value=True)

    device_options = ["CPU"]
    if cuda_count > 0:
        device_options.append("CUDA")
    device_label = st.selectbox("Compute device", device_options)

    cpu_threads = int(
        st.number_input(
            "CPU threads",
            min_value=1,
            max_value=max(1, logical_cpus),
            value=default_threads,
            disabled=device_label == "CUDA",
            help="CTranslate2 uses these OpenMP threads on CPU.",
        )
    )
    beam_size = int(
        st.selectbox(
            "Beam size",
            options=[1, 5],
            index=0,
            help="1 is much faster; 5 may improve accuracy slightly.",
        )
    )

    st.write(f"**Logical CPU threads:** {logical_cpus}")
    st.write(f"**CTranslate2 CUDA devices:** {cuda_count}")
    st.write(f"**Model:** `{MODEL_OPTIONS[model_label]}`")

    if ffmpeg_ready:
        st.success("FFmpeg detected")
        st.caption(ffmpeg_path)
    else:
        st.error("FFmpeg not detected")

if not ffmpeg_ready:
    st.error(
        "FFmpeg and FFprobe are required to normalise uploaded audio. "
        "Install FFmpeg and restart PowerShell before running this app."
    )

uploaded = st.file_uploader("Upload audio", type=AUDIO_TYPES)
if uploaded:
    data = uploaded.getvalue()
    st.audio(data)

    if st.button(
        "Transcribe",
        type="primary",
        use_container_width=True,
        disabled=not ffmpeg_ready,
    ):
        source_path: str | None = None
        wav_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded.name).suffix or ".audio",
            ) as temp:
                temp.write(data)
                source_path = temp.name

            duration = audio_duration(source_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_temp:
                wav_path = wav_temp.name

            with st.spinner("Decoding audio to 16 kHz mono WAV…"):
                convert_to_whisper_wav(source_path, wav_path)

            model_name = MODEL_OPTIONS[model_label]
            device = "cuda" if device_label == "CUDA" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            status = st.empty()
            progress = st.progress(0, text="Loading the transcription model…")
            started = time.perf_counter()

            model = load_whisper(model_name, device, compute_type, cpu_threads)
            language = LANGUAGES[language_label]
            task = "transcribe" if language == "en" else "translate"

            status.info(
                f"Transcribing on {device.upper()} with {compute_type}; "
                f"model={model_name}, CPU threads={cpu_threads if device == 'cpu' else 'n/a'}"
            )

            segments, info = model.transcribe(
                wav_path,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=False,
            )

            text_parts: list[str] = []
            total_duration = float(info.duration or duration or 0)
            for segment in segments:
                clean_segment = segment.text.strip()
                if clean_segment:
                    text_parts.append(clean_segment)
                if total_duration > 0:
                    percentage = min(100, max(0, int((segment.end / total_duration) * 100)))
                    progress.progress(
                        percentage,
                        text=f"Transcribing… {percentage}% ({segment.end / 60:.1f} min processed)",
                    )

            elapsed = time.perf_counter() - started
            progress.progress(100, text="Transcription complete")
            status.empty()

            transcript = " ".join(text_parts).strip()
            if normalise:
                transcript = uk_english(transcript)

            if not transcript:
                raise RuntimeError(
                    "No speech was transcribed. Check that the file contains audible speech "
                    "and select its spoken language explicitly."
                )

            st.session_state.transcript = transcript
            st.session_state.filename = output_name(uploaded.name)
            st.session_state.elapsed = elapsed
            st.session_state.duration = duration or info.duration
            st.session_state.runtime = f"{device.upper()} / {compute_type} / {model_name}"
            st.session_state.token_count = count_tokens(model, transcript)
            st.success("Transcription complete.")

        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            st.info(
                "For reliable CPU use, select Large v3 Turbo, CPU, 4–8 threads, "
                "and beam size 1. CUDA requires compatible NVIDIA runtime libraries."
            )
        finally:
            if source_path:
                Path(source_path).unlink(missing_ok=True)
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

if "transcript" in st.session_state:
    st.subheader("Transcript")
    text = st.text_area(
        "Review and edit",
        key="transcript",
        height=360,
        label_visibility="collapsed",
    )
    words = len(text.split())
    elapsed = float(st.session_state.get("elapsed", 0))
    duration = st.session_state.get("duration")

    columns = st.columns(4)
    columns[0].metric("Words", f"{words:,}")
    columns[1].metric("Text tokens", f"{int(st.session_state.get('token_count', 0)):,}")
    columns[2].metric("Runtime", f"{elapsed:.1f} s")
    columns[3].metric(
        "Real-time factor",
        f"{elapsed / duration:.2f}×" if duration else "Unknown",
    )

    st.caption(str(st.session_state.get("runtime", "")))
    st.caption(
        "The token count uses Whisper's tokenizer and is informational only. "
        "Local inference uses zero billable OpenAI API tokens."
    )
    st.download_button(
        "Download transcript (.txt)",
        data=text.encode("utf-8"),
        file_name=st.session_state.get("filename", "transcript_uk.txt"),
        mime="text/plain; charset=utf-8",
        type="primary",
        use_container_width=True,
    )
