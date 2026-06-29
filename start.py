#!/usr/bin/env python3
"""
🚀 Фабрика видео — запуск одной командой.

Сначала нужно задать ключ DeepSeek для генерации шуток:
  export DEEPSEEK_API_KEY="sk-..."
  
Если ключ не задан — берёт шутки из jokes.txt

Запуск:
  python start.py              # light
  python start.py --heavy      # heavy
  python start.py --ultra      # ultra
"""

import sys
from pathlib import Path

M = {
    "light": ["sd-v1-5", "silero_ru.pt"],
    "heavy": ["sdxl-base", "xtts-v2", "animatediff-motion"],
    "ultra": ["sdxl-base", "xtts-v2", "controlnet-canny", "wav2lip", "gfpgan"],
}

mode = "ultra" if "--ultra" in sys.argv else "heavy" if "--heavy" in sys.argv else "light"

md = Path(__file__).parent / "models"
missing = [n for n in M[mode] if not (md / n).exists()]
if missing:
    print(f"❌ Нет моделей: {', '.join(missing)}")
    print(f"Запусти сначала: python download_models.py --mode {mode}")
    sys.exit(1)

from factory import run_pipeline
run_pipeline(mode)
