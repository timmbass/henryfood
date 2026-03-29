"""Pi-side voice capture script.

Run with system Python (NOT inside a venv — gpiozero needs direct GPIO access):
    python3 capture.py

Flow:
    Press button  → start arecord
    Press button  → stop arecord, save WAV, rsync to i5, trigger pipeline via SSH
"""

from gpiozero import Button
import signal
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import time

button = Button(17, pull_up=True, bounce_time=0.2)

audio_dir = Path("/home/timothybass/apps/henryfood/data/audio_inbox")
audio_dir.mkdir(parents=True, exist_ok=True)

I5_HOST = "ubuntu-i5"
I5_USER = "timbass"
REMOTE_DIR = "/home/timbass/ai-lab/henryfood/data/audio_inbox"
REMOTE_SCRIPT = "/home/timbass/ai-lab/henryfood/scripts/process_audio_remote.sh"

recording_proc = None
current_file = None
_lock = threading.Lock()


def trigger_pipeline(remote_filepath: str) -> None:
    """SSH into i5 and kick off transcription → CSV append."""
    print(f"Triggering pipeline for {remote_filepath}...", flush=True)
    result = subprocess.run([
        "ssh",
        f"{I5_USER}@{I5_HOST}",
        f"bash {REMOTE_SCRIPT} {remote_filepath}",
    ], check=False)
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
    ], check=False)
    if result.returncode == 0:
        print("Transfer complete.", flush=True)
        trigger_pipeline(f"{REMOTE_DIR}/{filepath.name}")
    else:
        print("Transfer failed.", flush=True)


def toggle_recording() -> None:
    global recording_proc, current_file

    with _lock:
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

            # Bug 2: detect immediate failure (e.g. wrong device number)
            time.sleep(0.1)
            if recording_proc.poll() is not None:
                print(
                    f"arecord failed immediately (exit {recording_proc.returncode}) "
                    "— check the ALSA device (arecord -l)",
                    flush=True,
                )
                recording_proc = None
                current_file = None
                return
        else:
            print("Stopping recording", flush=True)
            recording_proc.send_signal(signal.SIGINT)
            try:
                recording_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                recording_proc.kill()
                try:
                    recording_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

            rc = recording_proc.returncode
            recording_proc = None

            # arecord exits with code 1 when stopped by SIGINT — this is normal
            # and the WAV file is still written correctly.  Only bail out on
            # unexpected codes (e.g. -9 for SIGKILL, 2+ for device errors).
            if rc not in (0, 1):
                print(f"arecord exited with error (code {rc}) — skipping transfer.", flush=True)
                return

            if not (current_file and current_file.exists() and current_file.stat().st_size > 44):
                print("Recording file missing or empty — skipping transfer.", flush=True)
                return

            print(f"Saved: {current_file}", flush=True)
            send_to_i5(current_file)


button.when_pressed = toggle_recording

print("Ready. Press once to start recording, again to stop.", flush=True)
signal.pause()
