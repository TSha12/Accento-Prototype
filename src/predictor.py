import librosa
import joblib
import whisper
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Load classifier and Whisper model once
    accent_clf = joblib.load("models/accent_classifier.pkl")
    logging.info("Accent classifier loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load accent classifier: {e}")
    accent_clf = None

try:
    whisper_model = whisper.load_model("tiny")
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

def predict_accent(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = mfcc.mean(axis=1).reshape(1, -1)
        accent = accent_clf.predict(features)[0]
        conf = max(accent_clf.predict_proba(features)[0])
        return accent, conf
    except Exception as e:
        logging.error(f"Error in predict_accent: {e}")
        return "unknown", 0.0

def accent_aware_stt(audio_file):
    try:
        accent, conf = predict_accent(audio_file)
        logging.info(f"Predicted accent: {accent} with confidence {conf}")

        if accent == "neutral":
            used_stt = "Google"
            transcript = "Simulated Google transcript"
        else:
            used_stt = "Whisper"
            if whisper_model is not None:
                try:
                    result = whisper_model.transcribe(audio_file)
                    transcript = result.get("text", "")
                except Exception as e:
                    logging.error(f"Whisper transcription failed: {e}")
                    transcript = "Error in transcription"
            else:
                transcript = "Whisper model not loaded"

        return {
            "accent": accent,
            "accent_confidence": float(conf),
            "used_stt": used_stt,
            "transcript": transcript
        }
    except Exception as e:
        logging.error(f"Error in accent_aware_stt: {e}")
        return {
            "accent": "error",
            "accent_confidence": 0.0,
            "used_stt": "error",
            "transcript": f"Error occurred: {str(e)}"
        }
