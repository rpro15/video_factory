# 🎬 Фабрика видео

AI-генератор мемных роликов. Работает полностью локально на GPU.

## Режимы

| Режим | GPU | Модели | Эффекты |
|-------|-----|--------|---------|
| light | GTX 1060 / RTX 2060 | ~2 GB | emoji, тряска, вспышки |
| heavy | RTX 3060+ | ~9 GB | AnimateDiff, XTTS-v2 |
| ultra | RTX 4090+ | ~20 GB | ControlNet, Wav2Lip, GFPGAN |

## Быстрый старт

```bash
# 1. Скачать модели
python download_models.py              # light
python download_models.py --mode heavy
python download_models.py --mode ultra

# 2. Запустить
python start.py              # light
python start.py --heavy      # heavy
python start.py --ultra      # ultra
```

Всё. Видео в `output/`.
