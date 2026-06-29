#!/usr/bin/env python3
"""
Скачивание моделей для Фабрики видео.
Запуск:
  python download_models.py              # light (~2 GB)
  python download_models.py --mode heavy # heavy (~9 GB)
  python download_models.py --mode ultra # ultra (~20 GB)
  python download_models.py --mode all   # всё
"""

import os, sys, argparse, subprocess
from pathlib import Path
import torch, shutil
from tqdm import tqdm
from huggingface_hub import snapshot_download

DIR = Path(__file__).parent / "models"
DIR.mkdir(exist_ok=True)

def hf(repo, name, desc=""):
    t = DIR / name
    if t.exists() and any(t.iterdir()):
        s = sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
        print(f"   ✅ {desc or name} ({s/1024/1024:.0f} MB)")
        return
    print(f"   📥 {desc or name}...")
    snapshot_download(repo_id=repo, local_dir=str(t), local_dir_use_symlinks=False, resume_download=True)
    print(f"   ✅ Готов")

def url(url, name, desc=""):
    t = DIR / name
    if t.exists():
        print(f"   ✅ {desc or name} ({t.stat().st_size/1024/1024:.0f} MB)")
        return
    print(f"   📥 {desc or name}...")
    t.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(url, str(t))
    print(f"   ✅ Готов")

def light():
    print("\n── LIGHT ──")
    hf("runwayml/stable-diffusion-v1-5", "sd-v1-5", "SD 1.5")
    url("https://models.silero.ai/models/tts/ru/v3_1_ru.pt", "silero_ru.pt", "Silero TTS")

def heavy():
    print("\n── HEAVY ──")
    hf("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base", "SDXL base")
    hf("coqui/XTTS-v2", "xtts-v2", "XTTS-v2")
    hf("guoyww/animatediff-motion-adapter-v1-5-2", "animatediff-motion", "AnimateDiff")

def ultra():
    print("\n── ULTRA ──")
    hf("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base", "SDXL base")
    hf("coqui/XTTS-v2", "xtts-v2", "XTTS-v2")
    hf("diffusers/controlnet-canny-sdxl-1.0", "controlnet-canny", "ControlNet Canny")
    hf("diffusers/controlnet-depth-sdxl-1.0", "controlnet-depth", "ControlNet Depth")
    hf("h94/IP-Adapter", "ip-adapter", "IP-Adapter")
    hf("guoyww/animatediff-motion-adapter-sdxl-beta", "animatediff-sdxl", "AnimateDiff SDXL")
    hf("ByteDance/SDXL-Lightning", "sdxl-lightning", "SDXL Lightning")
    print("   📥 Wav2Lip...")
    wl = DIR / "wav2lip"
    wl.mkdir(parents=True, exist_ok=True)
    if not (wl / "Wav2Lip").exists():
        subprocess.run(["git","clone","https://github.com/Rudrabha/Wav2Lip.git",str(wl/"Wav2Lip")], capture_output=True)
    url("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "gfpgan/GFPGANv1.4.pth", "GFPGAN")
    url("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.0/RealESRGAN_x4plus.pth", "realesrgan/RealESRGAN_x4plus.pth", "Real-ESRGAN")

def summary():
    print("\n" + "="*50)
    print(f"📁 models/:")
    total = 0
    for item in sorted(DIR.iterdir()):
        if item.is_dir():
            s = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
        else:
            s = item.stat().st_size
        if s > 10*1024*1024:
            print(f"   {item.name}: {s/1024/1024:.0f} MB")
        total += s
    print(f"━━━━━━━━━━━")
    print(f"📦 {total/1024/1024/1024:.1f} GB")
    print("="*50)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["light","heavy","ultra","all"], default="light")
    p.add_argument("--heavy", action="store_true")
    p.add_argument("--ultra", action="store_true")
    a = p.parse_args()
    mode = "ultra" if a.ultra else "heavy" if a.heavy else a.mode
    print(f"🎯 Режим: {mode}")
    light()
    if mode in ("heavy","ultra","all"): heavy()
    if mode in ("ultra","all"): ultra()
    summary()
