import os
import subprocess
import gradio as gr
from src.predictor import accent_aware_stt

# Set ffmpeg path
ffmpeg_path = r"F:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Optional: verify ffmpeg
try:
    subprocess.run([ffmpeg_path, "-version"], check=True)
    print("FFmpeg detected successfully!")
except FileNotFoundError:
    print("FFmpeg not found! Check the path.")

def transcribe(audio_file):
    if audio_file is None:
        return "No file uploaded"
    try:
        result = accent_aware_stt(audio_file)
        return f"Accent: {result['accent']} (conf: {result['accent_confidence']:.2f})\n" \
               f"Engine: {result['used_stt']}\n" \
               f"Transcript: {result['transcript']}"
    except Exception as e:
        return f"Error during processing: {str(e)}"

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(
        label="Result",
        lines=15,       # increase height
        max_lines=30,   # allow expansion
        placeholder="Result will appear here..."
    ),
    title="Accento: Accent-Aware Speech Recognition",
    description="Upload audio → detects accent → transcribes speech."
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
