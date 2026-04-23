#!/usr/bin/env python3
"""Batch process Chinese .mp4 files through pyvideotrans in Google Colab."""

import os
import re
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path("/content/input_videos")
OUTPUT_DIR = Path("/content/dubbed_output")
ERRORS_FILE = Path("/content/errors.log")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = list(INPUT_DIR.glob("*.mp4"))
if not files:
    print("Add your .mp4 files to /content/input_videos/ and run again")
    sys.exit(0)

def safe_name(name):
    base = Path(name).stem
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', base)
    return safe + ".mp4"

processed = skipped = failed = 0

for f in files:
    safe_input = safe_name(f.name)
    if safe_input != f.name:
        new_input = INPUT_DIR / safe_input
        f.rename(new_input)
        f = new_input
    
    output_file = OUTPUT_DIR / f"{f.stem}_dubbed.mp4"
    if output_file.exists():
        print(f"already done: {f.name}")
        skipped += 1
        continue
    
    print(f"Processing: {f.name}...")
    
    cmd = [
        "python", "cli.py",
        "--task", "vtv",
        "--name", str(f),
        "--target_dir", str(OUTPUT_DIR),
        "--source_language_code", "zh",
        "--target_language_code", "en",
        "--asr_name", "faster_whisper",
        "--whisper_model", "medium",
        "--translate_name", "google",
        "--tts_name", "edgetts",
        "--voice_name", "en-US-AriaNeural",
        "--subtitle_type", "1"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FAILED: {f.name}")
        with open(ERRORS_FILE, "a") as ef:
            ef.write(f"{f.name}: {result.stderr}\n")
        failed += 1
    else:
        print(f"Done: {f.name}")
        processed += 1

print(f"Done. {processed} processed, {skipped} skipped, {failed} failed. Check errors.log")