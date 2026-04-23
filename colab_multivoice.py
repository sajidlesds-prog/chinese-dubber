#!/usr/bin/env python3
"""Multi-voice Chinese to English video dubber with speaker diarization.

Features:
- Speaker diarization to detect different speakers
- Different English voices for different speakers
- Per-segment TTS at correct timestamps
- Audio timeline alignment
- Hardcoded subtitles
"""

import os
import sys
import subprocess
import json
import re
import time
import tempfile
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import files

INPUT_DIR = Path("/content/input_videos")
OUTPUT_DIR = Path("/content/dubbed_output")
TEMP_DIR = Path("/content/temp")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Voice mapping for different speaker types
SPEAKER_VOICES = {
    "male": "en-US-GuyNeural",      # Male adult
    "female": "en-US-AriaNeural",    # Female adult  
    "child": "en-US-SaraNeural",    # Child
    "elder": "en-US-EmmaNeural",    # Elder
    "default": "en-US-AriaNeural"
}

# Voice characteristics for detection
MALE_KEYWORDS = ["他", "他", "老公", "爸爸", "儿子", "哥", "先生", "男士", "男人"]
FEMALE_KEYWORDS = ["她", "她", "老婆", "妈妈", "女儿", "姐", "女士", "女人", "女孩"]
CHILD_KEYWORDS = ["小孩", "孩子", "小朋友", "宝宝", "童"]


@dataclass
class Segment:
    """Represents a subtitle segment with speaker info."""
    start: float
    end: float
    text: str
    text_en: str
    speaker: Optional[str] = None
    speaker_id: int = 0
    audio_file: Optional[str] = None
    

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    deps = [
        "faster-whisper",
        "edge-tts",
        "pydub",
        "soundfile",
        "numpy",
        "scipy"
    ]
    
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], capture_output=True)
    
    # ffmpeg
    subprocess.run(["apt-get", "install", "-qq", "ffmpeg"], capture_output=True)
    
    print("✅ Dependencies installed")


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video."""
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
          "-ar", "16000", "-ac", "1", "-y", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def detect_speaker_type(text: str) -> str:
    """Simple heuristic to detect speaker type from Chinese text context."""
    text = text[:100]  # Check first 100 chars
    
    # Count keywords
    male_score = sum(1 for w in MALE_KEYWORDS if w in text)
    female_score = sum(1 for w in FEMALE_KEYWORDS if w in text)
    child_score = sum(1 for w in CHILD_KEYWORDS if w in text)
    
    # Check for honorifics that indicate elder
    if any(w in text for w in ["爷爷", "奶奶", "叔叔", "阿姨", "老师", "老爷"]):
        return "elder"
    
    if child_score > 0:
        return "child"
    if male_score > female_score and male_score > 0:
        return "male"
    if female_score > male_score and female_score > 0:
        return "female"
    
    return "default"


def transcribe_with_diarization(audio_path: str, model_size: str = "medium") -> List[Segment]:
    """Transcribe audio with speaker detection."""
    from faster_whisper import WhisperModel
    
    print(f"Loading faster-whisper: {model_size}")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    print("Transcribing with VAD...")
    
    # Get segments with timestamps
    segments, info = model.transcribe(
        audio_path,
        language="zh",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True
    )
    
    # Group into speaker segments (simple clustering by timestamp gaps)
    results = []
    current_speaker = 0
    prev_end = 0
    gap_threshold = 2.0  # 2 second gap = new speaker
    
    segment_list = list(segments)
    
    for i, seg in enumerate(segment_list):
        # Skip short segments
        duration = seg.end - seg.start
        if duration < 0.5:
            continue
        
        # Detect new speaker by gap
        if seg.start - prev_end > gap_threshold:
            current_speaker = 1 - current_speaker
        
        # Detect speaker type from text
        speaker_type = detect_speaker_type(seg.text)
        
        results.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            text_en="",
            speaker=speaker_type,
            speaker_id=current_speaker
        ))
        
        prev_end = seg.end
    
    print(f"Found {len(results)} segments")
    return results


def translate_text(text: str) -> str:
    """Translate Chinese to English."""
    import requests
    
    if not text.strip():
        return ""
    
    # Chunk to avoid too long
    chunks = []
    current = ""
    for sentence in text.replace("\n", ". ").split("."):
        if len(current) + len(sentence) > 3500:
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
        
        time.sleep(0.8)  # Rate limiting
    
    return " ".join(results)


def generate_speech_async(text: str, output_path: str, voice: str = "en-US-AriaNeural", rate: str = "+12%"):
    """Generate speech using edge-tts."""
    import edge_tts
    
    async def generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
    
    asyncio.run(generate())


def generate_segment_audio(segments: List[Segment], temp_dir: Path):
    """Generate TTS for each segment with correct voice."""
    print("Generating audio for each segment...")
    
    # Group segments by speaker type for consistent voice
    speaker_voices = {}
    
    for i, seg in enumerate(segments):
        # Get voice for speaker type
        voice = SPEAKER_VOICES.get(seg.speaker, SPEAKER_VOICES["default"])
        
        # Alternate between similar speakers for same character consistency
        voice_key = (seg.speaker, seg.speaker_id)
        if voice_key not in speaker_voices:
            speaker_voices[voice_key] = voice
        
        output_file = temp_dir / f"seg_{i}.mp3"
        seg.audio_file = str(output_file)
        
        print(f"  Segment {i+1}: {seg.text_en[:40]}... ({seg.speaker})")
        
        generate_speech_async(seg.text_en, str(output_file), speaker_voices[voice_key])
    
    # Wait for all to complete
    time.sleep(1)


def create_subtitle_file(segments: List[Segment], output_path: str):
    """Create SRT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start_time = format_time(seg.start)
            end_time = format_time(seg.end)
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{seg.text_en}\n\n")


def format_time(seconds: float) -> str:
    """Format seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_dubbed_video(video_path: str, segments: List[Segment], output_path: str, temp_dir: Path):
    """Create dubbed video with proper timeline alignment."""
    
    # Step 1: Concatenate all segment audio with correct timing
    print("Creating audio timeline...")
    
    # Create file list for ffmpeg
    concat_file = temp_dir / "concat.txt"
    
    with open(concat_file, 'w') as f:
        # Build silence padding between segments
        for i, seg in enumerate(segments):
            if seg.audio_file:
                # Add silence before this segment
                if i == 0:
                    # First segment - start from beginning
                    f.write(f"file '{seg.audio_file}'\n")
                else:
                    prev_seg = segments[i-1]
                    silence_duration = seg.start - prev_seg.end
                    if silence_duration > 0:
                        # Create silence
                        silence_file = temp_dir / f"silence_{i}.mp3"
                        subprocess.run([
                            "ffmpeg", "-f", "lavfi", "-i", 
                            f"anullsrc=r=24000:cl=mono",
                            "-t", str(silence_duration),
                            "-y", str(silence_file)
                        ], capture_output=True)
                        f.write(f"file '{silence_file}'\n")
                    f.write(f"file '{seg.audio_file}'\n")
    
    # Concatenate audio
    combined_audio = temp_dir / "combined.mp3"
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy", "-y", str(combined_audio)
    ], capture_output=True)
    
    # Step 2: Replace audio in video
    print("Replacing audio in video...")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-i", str(combined_audio),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y", str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    
    # Step 3: Add hardcoded subtitles
    print("Adding subtitles...")
    subs_path = temp_dir / "subs.srt"
    create_subtitle_file(segments, str(subs_path))
    
    final_output = output_path.replace(".mp4", "_subs.mp4")
    cmd2 = [
        "ffmpeg", "-i", str(output_path),
        "-vf", f"subtitles={subs_path}",
        "-c:a", "copy",
        "-y", final_output
    ]
    subprocess.run(cmd2, capture_output=True)
    
    # Move final
    if Path(final_output).exists():
        Path(final_output).replace(output_path)
    
    return True


def dub_video(video_path: str, output_path: str):
    """Dub a single video file with multi-voice support."""
    print(f"\n{'='*50}")
    print(f"Processing: {Path(video_path).name}")
    print(f"{'='*50}")
    
    video_name = Path(video_path).stem
    temp_dir = TEMP_DIR / video_name
    temp_dir.mkdir(exist_ok=True)
    
    audio_path = temp_dir / "audio.wav"
    
    # Step 1: Extract audio
    print("\n[1/5] Extracting audio...")
    if not extract_audio(str(video_path), str(audio_path)):
        print("❌ Failed to extract audio")
        return False
    
    # Step 2: Transcribe with speaker detection
    print("\n[2/5] Transcribing with speaker detection...")
    segments = transcribe_with_diarization(str(audio_path))
    if not segments:
        print("❌ No speech detected")
        return False
    
    # Step 3: Translate
    print("\n[3/5] Translating to English...")
    for i, seg in enumerate(segments):
        print(f"  {i+1}/{len(segments)}: {seg.text[:30]}...")
        seg.text_en = translate_text(seg.text)
    
    # Step 4: Generate audio per segment
    print("\n[4/5] Generating voice audio...")
    generate_segment_audio(segments, temp_dir)
    
    # Wait for all audio to be generated
    time.sleep(2)
    
    # Step 5: Create final video
    print("\n[5/5] Creating dubbed video...")
    create_dubbed_video(video_path, segments, output_path, temp_dir)
    
    print(f"\n✅ Done: {output_path}")
    return True


def main():
    # Install deps if in Colab
    if IN_COLAB:
        install_dependencies()
    
    # Find videos
    videos = list(INPUT_DIR.glob("*.mp4"))
    
    if not videos:
        print("No .mp4 files found in /content/input_videos/")
        return
    
    print(f"Found {len(videos)} video(s)")
    
    # Process each video
    for video in videos:
        output_file = OUTPUT_DIR / f"{video.stem}_dubbed.mp4"
        
        if output_file.exists():
            print(f"Skipping (exists): {video.name}")
            continue
        
        dub_video(str(video), str(output_file))
    
    print("\n" + "="*50)
    print("COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    for f in OUTPUT_DIR.glob("*.mp4"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()