# Accento: Accent-Aware Speech Recognition

Accento bridges communication gaps caused by accent diversity by intelligently selecting the best transcription engine for users. It empowers inclusive technology and paves the way for better human-computer interaction across diverse linguistic backgrounds.
It detects the accent of a speaker and transcribes speech using a custom accent classifier and OpenAI’s Whisper providing a dynamic web interface via Gradio and a JSON API for programmatic use.

# Features
1.Detects speaker accent.
2.Transcribes speech to text.
3.Dynamic web UI with real-time results.
4.Returns structured JSON for API integration.
5.Supports any audio file compatible with FFmpeg.

# Tech Stack
| Component        | Technology / Library            |
| ---------------- | ------------------------------- |
| Backend          | Python 3.10+                    |
| Accent Detection | Custom Accent Classifier        |
| Speech-to-Text   | Whisper                         |
| Audio Processing | Librosa, FFmpeg                 |
| Web UI           | Gradio (Blocks)                 |
| API Access       | Gradio `/api/predict/` endpoint |

# Installation
git clone <repo-url>
cd Accento

# Activate venv
python -m venv venv

# Install Dependencies
pip install -r requirements.txt

# Set FFmpeg path in app.py:
ffmpeg_path = r"F:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Web UI
python app.py
Open http://127.0.0.1:7860
Upload audio → see Accent, Confidence, Engine, Transcript.

# API Access
Endpoint: http://127.0.0.1:7860/api/predict/

Use call_api.py for programmatic requests:

python call_api.py

## Repository notes (important)

- Large generated and dataset files are NOT tracked in this repository. The following paths are ignored via `.gitignore`:
	- `dataset/**/*.wav`, `dataset/**/*.mp3`, `dataset/**/*.flac` (audio dataset)
	- `data/*.npy` (precomputed features / saved arrays)
	- `models/*.pkl` (trained model files)
	- `plots/`, `.gradio/`
	- `venv/`, editor configs, caches, etc.

- I removed previously-committed large files from the index and committed an updated `.gitignore`. Those files remain on your local machine but will no longer be pushed to the remote in new commits.

## Restoring data / models locally

If you need to re-populate the repository on a fresh clone, place your files in the expected locations on your machine (these are not provided in the repo):

- Put audio files under `dataset/lang/` (or the `dataset/` path used by your scripts).
- Put precomputed arrays under `data/` (for example `features.npy`, `labels.npy`, `X_test.npy`, `y_test.npy`).
- Put trained models under `models/` (for example `models/accent_classifier.pkl`).

Your local copy will work with these files in-place even though they are ignored by Git.

## Quick start (Windows PowerShell)

```powershell
cd path\to\Accento
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
# Set ffmpeg path in app.py or ensure ffmpeg is on PATH
python app.py
# Open http://127.0.0.1:7860 in your browser
```

## API usage (programmatic)

Run the app and then use `call_api.py` to POST an audio file to the Gradio endpoint:

```powershell
python call_api.py
# Follow the prompt to enter the path to your audio file
```

## Notes about history and repo size

- Removing files from the index with `git rm --cached` prevents them from being included in future commits but does not remove them from prior commits. If you need the large files purged from repository history (to shrink remote size), we can run a history-rewrite using `git filter-repo` or the BFG Repo-Cleaner — this will rewrite commit hashes and requires coordination with anyone else using the repo.

If you'd like, I can add a small `DATA.md` with example commands to download or prepare the dataset outside Git and wire it into the project.
