"""Pi-side voice capture script.

Run with system Python (NOT inside a venv — gpiozero needs direct GPIO access):
    python3 capture.py

Flow:
    Press button  → start arecord
    Press button  → stop arecord, save WAV, rsync to i5, trigger pipeline via SSH
"""

from gpiozero import Button
from signal import pause
from pathlib import Path
from datetime import datetime
import subprocess

button = Button(17, pull_up=True, bounce_time=0.2)

audio_dir = Path("/home/timothybass/apps/henryfood/data/audio_inbox")
audio_dir.mkdir(parents=True, exist_ok=True)

I5_HOST = "ubuntu-i5"
I5_USER = "timbass"
REMOTE_DIR = "/home/timbass/ai-lab/henryfood/data/audio_inbox"
REMOTE_SCRIPT = "/home/timbass/ai-lab/henryfood/scripts/process_audio_remote.sh"

recording_proc = None
current_file = None


def trigger_pipeline(remote_filepath: str) -> None:
    """SSH into i5 and kick off transcription → CSV append."""
    print(f"Triggering pipeline for {remote_filepath}...", flush=True)
    result = subprocess.run([
        "ssh",
        f"{I5_USER}@{I5_HOST}",
        f"bash {REMOTE_SCRIPT} {remote_filepath}",
    ])
    if result.returncode == 0:
        print("Pipeline triggered.", flush=True)
    else:
        print(f"Pipeline trigger failed (exit {result.returncode}).", flush=True)


def send_to_i5(filepath: Path) -> None:
    print(f"Sending {filepath.name} to i5...", flush=True)
    result = subprocess.run([
        "rsync",
        "-av",
        str(filepath),
        f"{I5_USER}@{I5_HOST}:{REMOTE_DIR}/",
    ])
    if result.returncode == 0:
        print("Transfer complete.", flush=True)
        trigger_pipeline(f"{REMOTE_DIR}/{filepath.name}")
    else:
        print("Transfer failed.", flush=True)


def toggle_recording() -> None:
    global recording_proc, current_file

    if recording_proc is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_file = audio_dir / f"meal_note_{ts}.wav"

        print(f"Starting recording: {current_file}", flush=True)

        recording_proc = subprocess.Popen([
            "arecord",
            "-D", "plughw:3,0",
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            str(current_file),
        ])
    else:
        print("Stopping recording", flush=True)
        recording_proc.terminate()
        recording_proc.wait()
        recording_proc = None
        print(f"Saved: {current_file}", flush=True)

        if current_file and current_file.exists():
            send_to_i5(current_file)


button.when_pressed = toggle_recording

print("Ready. Press once to start recording, again to stop.", flush=True)
pause()
