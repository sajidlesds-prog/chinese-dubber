"""
Google Colab Setup Script for PyVideoTrans
Run this in a Colab cell: !python colab_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd, check=True):
    """Run shell command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def setup_colab():
    """Setup PyVideoTrans for Google Colab"""
    
    # Get project root
    ROOT_DIR = Path.cwd()
    print(f"Project root: {ROOT_DIR}")
    
    # 1. Install system dependencies (ffmpeg)
    print("\n=== Installing system dependencies ===")
    run_cmd("apt-get update -qq")
    run_cmd("apt-get install -y -qq ffmpeg")
    run_cmd("ffmpeg -version")
    
    # 2. Install Python dependencies
    print("\n=== Installing Python dependencies ===")
    
    # Core dependencies (without PySide6 GUI)
    core_deps = [
        "edge-tts",
        "aiohttp",
        "tenacity",
        "torch --index-url https://download.pytorch.org/whl/cpu",
        "transformers",
        "huggingface-hub",
        "librosa",
        "soundfile",
        "numpy",
        "requests",
        "faster-whisper",
        "openai-whisper",
        "pysrt",
        "simplejson",
        "datetime",
        "xmltodict"
    ]
    
    for dep in core_deps:
        run_cmd(f"pip install -q {dep}")
    
    # 3. Create necessary directories
    print("\n=== Creating directories ===")
    dirs = [
        "output",
        "tmp",
        "logs",
        "models",
        "ffmpeg"
    ]
    for d in dirs:
        Path(ROOT_DIR / d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")
    
    # 4. Set environment for CLI mode (no GUI)
    print("\n=== Setting up CLI mode ===")
    os.environ['PYVIDEOTRANS_LANG'] = 'en'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Disable Qt GUI
    
    # 5. Create minimal config for Colab
    print("\n=== Creating Colab config ===")
    import json
    
    config = {
        "homedir": str(ROOT_DIR / "output"),
        "lang": "en",
        "tts_type": 0,  # Edge TTS (free, works in Colab)
        "source_language": "en",
        "target_language": "zh-cn",
        "subtitle_type": 1,
        "voice_role": "zh-CN-XiaoxiaoNeural",  # Default Chinese voice
        "voice_rate": "0",
        "voice_autorate": True,
        "edge_tts_emotion_enabled": True,  # Enable emotional dubbing
        "model_name": "large-v3-turbo",
        "recogn_type": 0,
        "translate_type": 0,
    }
    
    config_path = ROOT_DIR / "cfg.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Created config: {config_path}")
    
    params = {
        "source_language": "en",
        "target_language": "zh-cn",
        "tts_type": 0,
        "voice_role": "zh-CN-XiaoxiaoNeural",
        "edge_tts_emotion_enabled": True,
    }
    
    params_path = ROOT_DIR / "params.json"
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"Created params: {params_path}")
    
    # 6. Test imports
    print("\n=== Testing imports ===")
    try:
        import videotrans
        print("✓ videotrans imported successfully")
        
        from videotrans.configure.config import tr, params, settings
        print("✓ config module loaded")
        
        from videotrans.util.sentiment import detect_sentiment
        print("✓ sentiment module loaded (emotional dubbing ready)")
        
        from videotrans.tts import EDGE_TTS
        print("✓ TTS module loaded")
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False
    
    print("\n=== Setup Complete! ===")
    print("\nTo use PyVideoTrans in Colab:")
    print("1. Upload your video file")
    print("2. Run CLI command:")
    print(f"   !python -c \"from videotrans.mainwin._actions import start_translate; start_translate('your_video.mp4')\"")
    print("\nOr use the CLI wrapper (see colab_run.py)")
    
    return True

if __name__ == "__main__":
    setup_colab()
