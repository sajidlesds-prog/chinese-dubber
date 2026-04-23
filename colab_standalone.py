#!/usr/bin/env python3
"""Standalone Chinese to English video dubber for Google Colab.

This script bypasses pyvideotrans complex dependencies and uses:
- faster-whisper for speech recognition
- Custom Google translator for translation  
- edge-tts for voice synthesis
- ffmpeg for video processing
"""

import os
import sys
import subprocess
import json
import re
import time
import tempfile
from pathlib import Path

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import files, drive

INPUT_DIR = Path("/content/input_videos")
OUTPUT_DIR = Path("/content/dubbed_output")
TEMP_DIR = Path("/content/temp")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def install_dependencies():
    """Install required dependencies in Colab."""
    print("Installing dependencies...")
    
    deps = [
        "faster-whisper",
        "edge-tts", 
        "requests",
        "tenacity",
        "pydub",
        "soundfile",
        "torch",
        "torchaudio",
    ]
    
    for dep in deps:
        subprocess.run(["pip", "install", "-q", dep], capture_output=True)
    
    # ffmpeg
    subprocess.run(["apt-get", "install", "-qq", "ffmpeg"], capture_output=True)
    
    print("✅ Dependencies installed")


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg."""
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
           "-ar", "16000", "-ac", "1", "-y", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def transcribe_audio(audio_path: str, model_size: str = "medium") -> list:
    """Transcribe audio using faster-whisper."""
    import torch
    from faster_whisper import WhisperModel
    
    # Auto-detect GPU vs CPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        compute_type = "int8"
        print("⚠️ Using CPU")
    
    print(f"Loading faster-whisper model: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print("Transcribing audio...")
    segments, info = model.transcribe(
        audio_path,
        language="zh",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300}
    )
    
    results = []
    for segment in segments:
        duration_ms = (segment.end - segment.start) * 1000
        if duration_ms < 500:  # Skip segments under 0.5s
            continue
        results.append({
            "text": segment.text.strip(),
            "start": segment.start,
            "end": segment.end
        })
    
    print(f"Transcribed {len(results)} segments")
    return results


def translate_text(text: str) -> str:
    """Translate Chinese to English using web scraping."""
    import requests
    
    if not text.strip():
        return ""
    
    # Chunk text to avoid too long
    chunks = []
    current = ""
    for sentence in text.replace("\n", ". ").split("."):
        if len(current) + len(sentence) > 4000:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current += "." + sentence
    if current:
        chunks.append(current.strip())
    
    results = []
    for chunk in chunks:
        url = f"https://translate.google.com/m?sl=zh-CN&tl=en&hl=en&q={chunk}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(2)
                else:
                    results.append(chunk)
                    continue
        
        match = re.search(r'<div\s+class=\Wresult-container\W>([^<]+?)<', response.text)
        if match:
            results.append(match.group(1))
        else:
            results.append(chunk)
        
        time.sleep(1)  # Rate limiting
    
    return " ".join(results)


def generate_speech(text: str, output_path: str, voice: str = "en-US-AriaNeural"):
    """Generate speech using edge-tts."""
    import edge_tts
    
    async def generate():
        communicate = edge_tts.Communicate(text, voice, rate="+12%")
        await communicate.save(output_path)
    
    import asyncio
    asyncio.run(generate())


def create_subtitle_file(segments: list, output_path: str):
    """Create SRT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start_time = format_time(seg['start'])
            end_time = format_time(seg['end'])
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{seg.get('text_en', seg['text'])}\n\n")


def format_time(seconds: float) -> str:
    """Format seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def dub_video(video_path: str, output_path: str):
    """Dub a single video file."""
    print(f"\n=== Processing: {Path(video_path).name} ===")
    
    video_name = Path(video_path).stem
    audio_path = TEMP_DIR / f"{video_name}_audio.wav"
    subs_path = TEMP_DIR / f"{video_name}.srt"
    
    # Step 1: Extract audio
    print("1. Extracting audio...")
    if not extract_audio(str(video_path), str(audio_path)):
        print("❌ Failed to extract audio")
        return False
    
    # Step 2: Transcribe
    print("2. Transcribing...")
    segments = transcribe_audio(str(audio_path))
    if not segments:
        print("❌ No speech detected")
        return False
    
    # Step 3: Translate
    print("3. Translating...")
    for seg in segments:
        print(f"  Translating: {seg['text'][:50]}...")
        seg['text_en'] = translate_text(seg['text'])
    
    # Step 4: Create subtitles
    print("4. Creating subtitles...")
    create_subtitle_file(segments, str(subs_path))
    
    # Step 5: Generate dubbed audio
    print("5. Generating English voice...")
    all_text = " ".join([seg['text_en'] for seg in segments])
    dub_audio_path = TEMP_DIR / f"{video_name}_dubbed.mp3"
    generate_speech(all_text, str(dub_audio_path))
    
    # Step 6: Combine with video
    print("6. Creating final video...")
    # Use ffmpeg to replace audio
    cmd = [
        "ffmpeg", "-i", video_path,
        "-i", str(dub_audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-y",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    
    print(f"✅ Done: {output_path}")
    return True


def main():
    # Install deps if in Colab
    if IN_COLAB:
        install_dependencies()
    
    # Find videos
    videos = list(INPUT_DIR.glob("*.mp4"))
    
    if not videos:
        print("No .mp4 files found in /content/input_videos/")
        print("Upload your video file and run again")
        return
    
    print(f"Found {len(videos)} video(s)")
    
    # Process each video
    for video in videos:
        output_file = OUTPUT_DIR / f"{video.stem}_dubbed.mp4"
        
        if output_file.exists():
            print(f"Skipping (already exists): {video.name}")
            continue
        
        success = dub_video(str(video), str(output_file))
        
        if not success:
            print(f"Failed: {video.name}")
    
    print("\n=== Complete ===")
    print(f"Output files in: {OUTPUT_DIR}")
    
    # List output files
    for f in OUTPUT_DIR.glob("*.mp4"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()