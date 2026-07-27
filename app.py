from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Any

import librosa
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "openai/whisper-large-v3"
AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "aac"]
LANGUAGES: dict[str, str | None] = {
    "English (default)": "english",
    "Auto-detect": None,
    "Arabic": "arabic",
    "Chinese": "chinese",
    "Dutch": "dutch",
    "French": "french",
    "German": "german",
    "Hindi": "hindi",
    "Italian": "italian",
    "Japanese": "japanese",
    "Korean": "korean",
    "Marathi": "marathi",
    "Portuguese": "portuguese",
    "Spanish": "spanish",
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
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(UK_MAP, key=len, reverse=True))) + r")\b", re.I)

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


@st.cache_resource(show_spinner=False)
def load_whisper() -> tuple[Any, str]:
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda:0" if use_cuda else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        torch_dtype=dtype,
        device=0 if use_cuda else -1,
    )
    return asr, device


st.set_page_config(page_title="Local Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Local Whisper Transcriber")
st.caption("Local, private transcription with openai/whisper-large-v3. No paid transcription API is used.")

with st.sidebar:
    language_label = st.selectbox("Spoken language", list(LANGUAGES))
    normalise = st.checkbox("Apply conservative UK spelling", value=True)
    batch_size = st.number_input(
        "Inference batch size",
        min_value=1,
        max_value=16,
        value=4 if torch.cuda.is_available() else 1,
        help="Use 1 on CPU or if GPU memory is limited.",
    )
    st.write(f"**Model:** `{MODEL_ID}`")
    st.write("**Runtime:** " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    st.caption("The model downloads to the Hugging Face cache on first use.")

uploaded = st.file_uploader("Upload audio", type=AUDIO_TYPES)
if uploaded:
    data = uploaded.getvalue()
    st.audio(data)

    if st.button("Transcribe", type="primary", use_container_width=True):
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".audio") as temp:
                temp.write(data)
                temp_path = temp.name

            try:
                duration = float(librosa.get_duration(path=temp_path))
            except Exception:
                duration = None

            with st.spinner("Loading Whisper and transcribing locally…"):
                started = time.perf_counter()
                asr, device = load_whisper()
                language = LANGUAGES[language_label]
                task = "transcribe" if language == "english" else "translate"
                generate_kwargs: dict[str, str] = {"task": task}
                if language:
                    generate_kwargs["language"] = language
                result = asr(
                    temp_path,
                    batch_size=int(batch_size),
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                )
                elapsed = time.perf_counter() - started

            transcript = str(result.get("text", "")).strip()
            if normalise:
                transcript = uk_english(transcript)

            st.session_state.transcript = transcript
            st.session_state.filename = output_name(uploaded.name)
            st.session_state.elapsed = elapsed
            st.session_state.duration = duration
            st.session_state.device = device
            st.success("Transcription complete.")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            st.error("GPU memory is insufficient. Set batch size to 1, close GPU-heavy programs, or use CPU PyTorch.")
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            st.info("Some formats require FFmpeg. Also confirm that the correct PyTorch CPU or CUDA build is installed.")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

if "transcript" in st.session_state:
    st.subheader("Transcript")
    text = st.text_area("Review and edit", key="transcript", height=360, label_visibility="collapsed")
    asr, _ = load_whisper()
    token_count = len(asr.tokenizer.encode(text, add_special_tokens=False)) if text else 0
    words = len(text.split())
    elapsed = float(st.session_state.get("elapsed", 0))
    duration = st.session_state.get("duration")

    columns = st.columns(4)
    columns[0].metric("Words", f"{words:,}")
    columns[1].metric("Text tokens", f"{token_count:,}")
    columns[2].metric("Runtime", f"{elapsed:.1f} s")
    columns[3].metric("Real-time factor", f"{elapsed / duration:.2f}×" if duration else "Unknown")

    st.caption("The token count uses Whisper's tokenizer and is informational only. Local inference uses zero billable OpenAI API tokens.")
    st.download_button(
        "Download transcript (.txt)",
        data=text.encode("utf-8"),
        file_name=st.session_state.get("filename", "transcript_uk.txt"),
        mime="text/plain; charset=utf-8",
        type="primary",
        use_container_width=True,
    )
