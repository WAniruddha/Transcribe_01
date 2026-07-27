# Local Whisper Transcriber

A basic Streamlit application for local audio transcription using Hugging Face Transformers and `openai/whisper-large-v3`.

The app:

- uploads common audio or audio-containing media formats;
- defaults to English transcription;
- translates explicitly selected non-English or auto-detected speech into English;
- applies a small, conservative UK-English spelling normalisation pass;
- keeps the transcript editable before download;
- downloads the final result as a UTF-8 `.txt` file;
- displays word count, approximate Whisper text-token count, runtime and real-time factor;
- runs locally, without a paid OpenAI transcription API.

## Important hardware note

`whisper-large-v3` is a large model. Its model weights are approximately 3.09 GB, and inference requires additional RAM or VRAM. An NVIDIA GPU is strongly preferable. CPU execution is supported by the app but may be very slow for long recordings.

The first run downloads the model into the Hugging Face cache. It is not stored in this Git repository.

## Windows setup using your existing E1 virtual environment

Target locations:

```text
Virtual environment: D:\02_Applications\10_VEnv\E1
Project parent:      D:\06_UpSkill\04_AI_Outskill\Projects
Repository folder:  D:\06_UpSkill\04_AI_Outskill\Projects\Transcribe_01
```

### 1. Clone the repository

Open PowerShell:

```powershell
Set-Location "D:\06_UpSkill\04_AI_Outskill\Projects"
git clone https://github.com/WAniruddha/Transcribe_01.git
Set-Location ".\Transcribe_01"
```

### 2. Confirm the E1 Python version

```powershell
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" --version
```

Use a currently supported Python version for your selected PyTorch build.

### 3. Install PyTorch

First check whether E1 already has a working PyTorch installation:

```powershell
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If PyTorch is missing, use the official PyTorch **Start Locally** selector to choose Windows, Pip and either CPU or the CUDA build appropriate for your NVIDIA driver:

- https://pytorch.org/get-started/locally/

A CPU-only installation commonly uses:

```powershell
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA, copy the current command from the official selector rather than guessing the CUDA wheel.

### 4. Install the app dependencies

```powershell
.\setup_windows.ps1
```

Equivalent manual command:

```powershell
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -m pip install -r requirements.txt
```

### 5. Install FFmpeg when needed

WAV, FLAC and many MP3 files may work directly. Formats such as M4A, WebM, MP4 and AAC commonly need FFmpeg available on `PATH`.

Download information:

- https://ffmpeg.org/download.html

Verify:

```powershell
ffmpeg -version
```

### 6. Run the Streamlit app

```powershell
.\run_app.ps1
```

Or:

```powershell
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -m streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`.

## How language handling works

- **English (default):** direct English transcription with `task="transcribe"` and `language="english"`.
- **Auto-detect:** Whisper detects the source and uses `task="translate"`, producing English text.
- **Named non-English language:** the chosen source language is supplied to Whisper and translated into English.

The UK-English pass is intentionally conservative. It changes a limited set of common spellings such as `color` to `colour`, while avoiding many context-sensitive pairs. Always review names, technical terms and punctuation in the editable text box.

## Token counting and cost

This project makes no OpenAI API call, so it uses **zero billable OpenAI API tokens**.

The app displays an approximate **Whisper text-token count** by applying the model's tokenizer to the finished transcript. This is useful for estimating the transcript size before sending it to another language model, but it is not a billing figure and does not represent audio-input tokens.

For a separate OpenAI API application, token usage is normally read from the API response's `usage` field or from the OpenAI usage dashboard. That is intentionally not included here because this application is fully local.

## Project structure

```text
Transcribe_01/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies other than PyTorch
├── setup_windows.ps1       # Installs dependencies into E1 and validates syntax
├── run_app.ps1             # Runs Streamlit using E1 directly
├── AI_BUILD_PROMPT.md      # Reusable Codex/AI development prompt
├── README.md
└── .gitignore
```

## Basic validation

```powershell
& "D:\02_Applications\10_VEnv\E1\Scripts\python.exe" -m py_compile app.py
```

Then run the UI and test with a short English WAV or MP3 before attempting a long recording.

## Privacy

Audio is written to a temporary local file only for transcription and is deleted when the operation finishes. The uploaded audio and generated transcript are not sent to an OpenAI transcription API by this code. Model files are downloaded from Hugging Face on first use.
