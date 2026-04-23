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

# Extended voice lists for drama with many characters
# Up to 25 unique male voices and 20 unique female voices
MALE_VOICES = [
    "en-US-GuyNeural",      # Deep male
    "en-US-RyanNeural",    # Medium male  
    "en-US-DavisNeural",   # Professional
    "en-US-AdamNeural",   # Young adult
    "en-US-BrandonNeural",  # Deep
    "en-US-ChristopherNeural", # Formal
    "en-US-CoryNeural",    # Casual
    "en-US-DanielNeural",  # Standard
    "en-US-DerekNeural",  # Deep
    "en-US-DEVINNeural",   # Youthful
    "en-US-ELIJAHNeural", # Deep
    "en-US-EthanNeural",   # Young
    "en-US-FranklinNeural", # Mature
    "en-US-FredNeural",   # Deep
    "en-US-GregoryNeural", # Formal
    "en-US-JacobNeural",  # Young
    "en-US-JasonNeural", # Standard
    "en-US-JeffNeural",  # Deep
    "en-US-JeremyNeural", # Casual
    "en-US-JoeNeural",   # Deep
    "en-US-MarcusNeural", # Deep
    "en-US-MarkNeural",  # Standard
    "en-US-NathanNeural", # Deep
    "en-US-PhillipNeural", # Formal
    "en-US-RichardNeural"  # Mature
]

FEMALE_VOICES = [
    "en-US-AriaNeural",     # Standard female
    "en-US-JennyNeural",    # Medium
    "en-US-SaraNeural",    # Light
    "en-US-AdaNeural",     # Young
    "en-US-AshleyNeural", # Casual
    "en-US-AudraNeural",   # Deep
    "en-US-BarbaraNeural", # Mature
    "en-US-BriannaNeural", # Young
    "en-US-CamilaNeural", # Youthful
    "en-US-CarolNeural",   # Standard
    "en-US-CrystalNeural", # Deep
    "en-US-DonnaNeural",   # Mature
    "en-US-EmilyNeural",  # Young
    "en-US-EvaNeural",    # Light
    "en-US-FionaNeural",   # Standard
    "en-US-GraceNeural",  # Light
    "en-US-HeatherNeural", # Standard
    "en-US-IsabellaNeural", # Youthful
    "en-US-JaneNeural",   # Formal
    "en-US-JuliaNeural"  # Standard
]

# Fallback voices for when we exceed the list
MALE_VOICES.extend(["en-US-GuyNeural"] * 10)  # Repeat if needed
FEMALE_VOICES.extend(["en-US-AriaNeural"] * 5)

# English name lists for character name assignment
MALE_NAMES = [
    "Ethan", "James", "Michael", "Daniel", "William", 
    "Alexander", "Benjamin", "Matthew", "David", "Joseph",
    "Andrew", "Ryan", "John", "Christopher", "Kevin",
    "Thomas", "Brandon", "Tyler", "Zachary", "Brian",
    "Eric", "Aaron", "Justin", "Jason", "Adam"
]

FEMALE_NAMES = [
    "Emma", "Olivia", "Sophia", "Ava", "Isabella",
    "Mia", "Charlotte", "Amelia", "Harper", "Evelyn",
    "Abigail", "Emily", "Elizabeth", "Sofia", "Avery",
    "Ella", "Scarlett", "Grace", "Chloe", "Victoria"
]

# Name mapping: tracks Chinese name -> English name (consistent throughout drama)
CHARACTER_NAMES = {}  # {speaker_id: english_name}

def assign_english_name(speaker_id: int, speaker_type: str, chinese_text: str = "") -> str:
    """Assign consistent English name to a Chinese character."""
    global CHARACTER_NAMES
    
    if speaker_id in CHARACTER_NAMES:
        return CHARACTER_NAMES[speaker_id]
    
    # Select name based on speaker type and ID
    if speaker_type in ["male", "elder"]:
        name = MALE_NAMES[speaker_id % len(MALE_NAMES)]
    else:
        name = FEMALE_NAMES[speaker_id % len(FEMALE_NAMES)]
    
    CHARACTER_NAMES[speaker_id] = name
    return name

# Keywords for speaker detection
MALE_KEYWORDS = ["他", "老公", "爸爸", "儿子", "哥", "先生", "男士", "男人", "丈夫", "父亲", "男孩", "小哥", "老"]
FEMALE_KEYWORDS = ["她", "老婆", "妈妈", "女儿", "姐", "女士", "女人", "妻子", "母亲", "女孩", "小姐", "姑"]
CHILD_KEYWORDS = ["小孩", "孩子", "小朋友", "宝宝", "童", "小孩", "娃"]
ELDER_KEYWORDS = ["爷爷", "奶奶", "叔叔", "阿姨", "老师", "老爷", "老", "爷爷", "奶奶"]


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
    voice: Optional[str] = None
    

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
    """Detect speaker type from Chinese text context."""
    text = text[:150]  # Check more context
    
    # Check for elder first (most distinct)
    if any(w in text for w in ELDER_KEYWORDS):
        return "elder"
    
    # Check for child
    if any(w in text for w in CHILD_KEYWORDS):
        return "child"
    
    # Count keywords for male/female
    male_score = sum(1 for w in MALE_KEYWORDS if w in text)
    female_score = sum(1 for w in FEMALE_KEYWORDS if w in text)
    
    if female_score > male_score and female_score > 0:
        return "female"
    if male_score > female_score and male_score > 0:
        return "male"
    
    return "default"


def get_voice_for_speaker(speaker_type: str, speaker_id: int) -> str:
    """Get appropriate voice based on speaker type and ID.
    
    For same gender speakers, alternates between different voice tones
    to distinguish them.
    """
    if speaker_type == "elder":
        return "en-US-EmmaNeural"
    if speaker_type == "child":
        return "en-US-SaraNeural"
    if speaker_type == "male":
        # Alternate between different male voices
        return MALE_VOICES[speaker_id % len(MALE_VOICES)]
    if speaker_type == "female":
        # Alternate between different female voices  
        return FEMALE_VOICES[speaker_id % len(FEMALE_VOICES)]
    
    return "en-US-AriaNeural"


def transcribe_with_diarization(audio_path: str, model_size: str = "medium") -> List[Segment]:
    """Transcribe audio with consistent speaker tracking throughout video."""
    import torch
    from faster_whisper import WhisperModel
    
    # Auto-detect GPU vs CPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"  # Faster on GPU
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        compute_type = "int8"
        print("⚠️ Using CPU (GPU not available)")
    
    print(f"Loading faster-whisper: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print("Transcribing with VAD...")
    
    # Get segments with timestamps
    segments, info = model.transcribe(
        audio_path,
        language="zh",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True
    )
    
    # Group into speaker segments - CONSISTENT tracking throughout video
    results = []
    speaker_tracking = {}  # speaker_id -> speaker_type (keeps same voice for same person)
    next_speaker_id = 0
    
    segment_list = list(segments)
    
    for i, seg in enumerate(segment_list):
        # Skip short segments
        duration = seg.end - seg.start
        if duration < 0.5:
            continue
        
        # Detect speaker type from text
        speaker_type = detect_speaker_type(seg.text)
        
        # Check if this segment belongs to an existing speaker based on:
        # 1. Recent proximity (within 3 seconds = same speaker)
        # 2. Same speaker type (both male/female)
        
        assigned_speaker_id = None
        
        # Look at recent segments to find matching speaker
        for j in range(len(results)-1, max(len(results)-5, -1), -1):
            if j >= 0:
                prev = results[j]
                # If within 3 seconds and same type, likely same speaker
                if seg.start - prev.end < 3.0 and prev.speaker == speaker_type:
                    assigned_speaker_id = prev.speaker_id
                    break
        
        # If no match found, assign new speaker ID
        if assigned_speaker_id is None:
            # Check if we already have this speaker type tracked
            for sid, stype in speaker_tracking.items():
                if stype == speaker_type:
                    assigned_speaker_id = sid
                    break
        
        if assigned_speaker_id is None:
            assigned_speaker_id = next_speaker_id
            next_speaker_id += 1
            speaker_tracking[assigned_speaker_id] = speaker_type
        
        results.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            text_en="",
            speaker=speaker_type,
            speaker_id=assigned_speaker_id
        ))
    
    print(f"Found {len(results)} segments, {len(speaker_tracking)} unique speaker(s)")
    return results
    
    print(f"Found {len(results)} segments")
    return results


def translate_text(text: str) -> str:
    """Translate Chinese to English with cleanup."""
    import requests
    
    if not text.strip():
        return ""
    
    # Clean up text before translation
    # Remove excessive punctuation and normalize
    text = re.sub(r'[。！？]{2,}', '！', text)  # Multiple punctuation to single
    text = re.sub(r'[,，]{2,}', '，', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = text.strip()
    
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
            translated = match.group(1)
            # Clean up translation
            translated = translated.strip()
            # Fix common translation artifacts
            translated = re.sub(r'\s+', ' ', translated)  # Multiple spaces to single
            results.append(translated)
        else:
            results.append(chunk)
        
        time.sleep(0.8)  # Rate limiting
    
    final = " ".join(results)
    
    # Final cleanup
    final = re.sub(r'\s+', ' ', final)  # Normalize spaces
    final = final.strip()
    
    return final


def generate_speech_async(text: str, output_path: str, voice: str = "en-US-AriaNeural", rate: str = "+12%"):
    """Generate speech using edge-tts."""
    import edge_tts
    
    async def generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
    
    asyncio.run(generate())


def generate_segment_audio(segments: List[Segment], temp_dir: Path):
    """Generate TTS for each segment with correct voice and name."""
    print("Generating audio for each segment...")
    
    # Track voice and name per speaker for consistency
    speaker_voice_map = {}  # speaker_id -> voice
    speaker_name_map = {}   # speaker_id -> english_name
    
    for i, seg in enumerate(segments):
        # Get unique voice and name based on speaker ID
        # This ensures same character keeps same voice + name throughout
        if seg.speaker_id not in speaker_voice_map:
            speaker_voice_map[seg.speaker_id] = get_voice_for_speaker(seg.speaker, seg.speaker_id)
            speaker_name_map[seg.speaker_id] = assign_english_name(seg.speaker_id, seg.speaker)
        
        voice = speaker_voice_map[seg.speaker_id]
        name = speaker_name_map[seg.speaker_id]
        seg.voice = voice
        
        output_file = temp_dir / f"seg_{i}.mp3"
        seg.audio_file = str(output_file)
        
        print(f"  {i+1}: [{name}] {seg.text_en[:30]}...")
        
        generate_speech_async(seg.text_en, str(output_file), voice)
    
    print(f"\nCharacters: {speaker_name_map}")
    # Wait for all TTS to complete
    time.sleep(2)


def create_subtitle_file(segments: List[Segment], output_path: str):
    """Create SRT subtitle file with character names."""
    # Build name map
    name_map = {}
    for seg in segments:
        if seg.speaker_id not in name_map:
            name_map[seg.speaker_id] = assign_english_name(seg.speaker_id, seg.speaker)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start_time = format_time(seg.start)
            end_time = format_time(seg.end)
            name = name_map.get(seg.speaker_id, "Speaker")
            # Format: CharacterName: Translated text
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"[{name}]: {seg.text_en}\n\n")


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
    global CHARACTER_NAMES
    CHARACTER_NAMES = {}  # Reset name mapping for new video
    
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