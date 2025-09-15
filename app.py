import os
import subprocess
import gradio as gr
from src.predictor import accent_aware_stt

# ----------------------
# FFmpeg setup
# ----------------------
ffmpeg_path = r"F:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

try:
    subprocess.run([ffmpeg_path, "-version"], check=True)
    print("FFmpeg detected successfully!")
except FileNotFoundError:
    print("FFmpeg not found! Check the path.")

# ----------------------
# Accent-aware transcription function
# Returns structured dict
# ----------------------
def transcribe(audio_file):
    if audio_file is None:
        return {"error": "No file uploaded"}
    try:
        result = accent_aware_stt(audio_file)
        return {
            "Accent": result["accent"],
            "Confidence": round(result["accent_confidence"], 2),
            "Engine": result["used_stt"],
            "Transcript": result["transcript"]
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------
# Format output for UI display
# ----------------------
def format_output(audio_file):
    result = transcribe(audio_file)
    # Build a multi-line string for nice UI display
    if "error" in result:
        return result["error"]
    return (
        f"Accent: {result['Accent']}\n"
        f"Confidence: {result['Confidence']}\n"
        f"Engine: {result['Engine']}\n"
        f"Transcript:\n{result['Transcript']}"
    )

# ----------------------
# Gradio Blocks UI
# ----------------------
with gr.Blocks() as demo:
    gr.Markdown("# Accento: Accent-Aware Speech Recognition")
    gr.Markdown("Upload audio → detects accent → transcribes speech.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        output_box = gr.Textbox(
            label="Result",
            lines=15,
            max_lines=30,
            placeholder="Result will appear here...",
            interactive=False
        )

    # Update output dynamically when user uploads audio
    audio_input.change(fn=format_output, inputs=audio_input, outputs=output_box)

# ----------------------
# Launch server with web UI + API
# ----------------------
if __name__ == "__main__":
    demo.launch(
        share=True,            # Optional: public URL for external access
        server_name="0.0.0.0",
        server_port=7860,
    )
