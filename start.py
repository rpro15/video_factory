#!/usr/bin/env python3
"""🚀 Запуск: python start.py [--heavy|--ultra]"""
import sys
from pathlib import Path

M = {"light":["sd-v1-5","silero_ru.pt"], "heavy":["sdxl-base","xtts-v2","animatediff-motion"],
     "ultra":["sdxl-base","xtts-v2","controlnet-canny","wav2lip","gfpgan"]}
mode = "ultra" if "--ultra" in sys.argv else "heavy" if "--heavy" in sys.argv else "light"

md = Path(__file__).parent/"models"
miss = [n for n in M[mode] if not (md/n).exists()]
if miss:
    print(f"❌ Нет моделей: {', '.join(miss)}")
    print(f"Запусти: python download_models.py --mode {mode}")
    sys.exit(1)

from factory import run_pipeline
run_pipeline(mode)
