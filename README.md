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
