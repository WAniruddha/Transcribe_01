# Reusable Codex Prompt: Local Whisper Streamlit Transcriber

You are building and validating a Windows-first local speech-transcription project.

## Repository and environment

- GitHub repository: `https://github.com/WAniruddha/Transcribe_01.git`
- Local project directory: `D:\06_UpSkill\04_AI_Outskill\Projects\Transcribe_01`
- Existing virtual environment: `D:\02_Applications\10_VEnv\E1`
- Use PowerShell-compatible commands.
- Do not create another virtual environment unless E1 is unusable and the user explicitly approves it.

## Goal

Build a basic Streamlit UI that transcribes uploaded audio locally with Hugging Face model `openai/whisper-large-v3`. English is the default source language. For auto-detected or explicitly selected non-English audio, translate the speech into English. The output must be editable, cautiously normalised to UK English, and downloadable as a UTF-8 text file.

Do not use the OpenAI transcription API, paid inference endpoints, hosted Hugging Face inference, or API keys.

## Required behaviour

1. Provide a clean Streamlit page with file upload and audio preview.
2. Accept WAV, MP3, M4A, FLAC, OGG, WebM, MP4 and AAC, noting that FFmpeg may be required.
3. Use `transformers`, `torch`, `AutoModelForSpeechSeq2Seq`, `AutoProcessor` and the ASR pipeline.
4. Fix the model ID to `openai/whisper-large-v3`.
5. Detect CUDA with `torch.cuda.is_available()`.
6. Use float16 on CUDA and float32 on CPU.
7. Load with `low_cpu_mem_usage=True` and `use_safetensors=True`.
8. Cache the model/pipeline using `st.cache_resource`.
9. Use 30-second chunking for long recordings.
10. For English, use `task="transcribe"` and `language="english"`.
11. For auto-detect or a selected non-English language, use `task="translate"` so output is English.
12. Add an optional, conservative US-to-UK spelling conversion. Do not broadly rewrite the transcript.
13. Show the transcript in an editable text area.
14. Download the edited result as a `.txt` file with a safe filename based on the uploaded filename.
15. Display word count, approximate Whisper text-token count, runtime, audio duration/real-time factor where available, and runtime device.
16. Explain that local inference uses zero billable OpenAI API tokens. Token counts are informational only.
17. Save uploads to temporary files while preserving extensions and delete them in a `finally` block.
18. Handle CUDA out-of-memory and general decoding failures with useful messages.
19. Never commit audio, transcripts, model cache, secrets or virtual-environment files.

## Required files

- `app.py`
- `requirements.txt`, excluding PyTorch because CPU/CUDA wheels differ
- `README.md`
- `.gitignore`
- `setup_windows.ps1`
- `run_app.ps1`
- `AI_BUILD_PROMPT.md`

## Commands to document

```powershell
Set-Location "D:\06_UpSkill\04_AI_Outskill\Projects"
git clone https://github.com/WAniruddha/Transcribe_01.git
Set-Location ".\Transcribe_01"

& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -m pip install -r requirements.txt
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -m streamlit run app.py
```

Do not guess an NVIDIA CUDA wheel. Direct the user to the official PyTorch Start Locally selector. A CPU-only command may be shown separately.

## Validation

- Keep the implementation readable and beginner-friendly.
- Use type hints for non-trivial functions.
- Avoid Docker, databases and cloud services.
- Run `python -m py_compile app.py` with E1.
- When dependencies and hardware allow, launch Streamlit and smoke-test a short audio file.
- Check Git status and ensure no audio, model files, transcript files or secrets are staged.
- Report any validation that could not be performed, especially full `whisper-large-v3` inference when RAM, VRAM or model-download access is unavailable.

## Acceptance criteria

The user can clone the repository, install the correct PyTorch build and dependencies into E1, run Streamlit, upload English audio, obtain an editable UK-English transcript, view informational performance/size metrics, and download a text file without any paid transcription API call.
