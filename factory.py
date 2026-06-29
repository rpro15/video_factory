#!/usr/bin/env python3
"""
🎬 Фабрика видео — генератор мемных роликов.

Полностью автоматический режим:
1. AI генерирует 10 случайных шуток
2. Ты выбираешь номер
3. Фабрика делает видео

Запасной вариант: если нет интернета — берёт из jokes.txt

Запуск:
  python start.py              # light
  python start.py --heavy      # heavy
  python start.py --ultra      # ultra
"""

import os, sys, re, random, argparse, logging, datetime, math, json
from pathlib import Path
import yaml, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip, concatenate_videoclips, VideoFileClip
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── КОНФИГ ───

def load_config(mode):
    p = Path(__file__).parent / f"config_{mode}.yaml"
    if not p.exists():
        log.error(f"Нет конфига: {p}")
        sys.exit(1)
    c = yaml.safe_load(open(p))
    base = Path(__file__).parent
    for k in ("models_dir","output_dir","temp_dir"):
        if c.get(k) and not Path(c[k]).is_absolute():
            c[k] = str(base / c[k])
    for k in ("sd_model","tts_path","face_image","wav2lip_path","gfpgan_path","realesrgan_path",
              "ip_adapter_path","animatediff_motion","animatediff_lightning"):
        if c.get(k) and not Path(c[k]).is_absolute():
            c[k] = str(base / c[k])
    log.info(f"Конфиг: {p.name}")
    return c


# ─── ГЕНЕРАЦИЯ ШУТОК ЧЕРЕЗ AI ───

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

def generate_jokes_ai(count=10):
    """
    Генерирует шутки через DeepSeek API.
    Если ключ не задан или API недоступен — возвращает None.
    """
    if not DEEPSEEK_API_KEY:
        log.info("ℹ️  DEEPSEEK_API_KEY не задан. Использую jokes.txt")
        return None
    
    log.info(f"🤖 Генерация {count} шуток через DeepSeek...")
    try:
        prompt = (
            f"Придумай {count} коротких абсурдных шуток или анекдотов на русском языке. "
            f"Каждая шутка — одно предложение до 100 символов. "
            f"Ответ строго в формате JSON: {{\"jokes\": [\"шутка1\", \"шутка2\", ...]}}"
        )
        
        req = Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=json.dumps({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.9,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            }
        )
        
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            
            # Парсим JSON из ответа
            try:
                jokes = json.loads(content)["jokes"]
            except:
                # Если не JSON — вытаскиваем строки
                jokes = [l.strip("- ").strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("{")]
            
            if jokes and len(jokes) >= count:
                log.info(f"✅ Сгенерировано {len(jokes)} шуток")
                return jokes[:count]
        
        log.warning("⚠️ Не удалось распарсить ответ AI")
        return None
        
    except Exception as e:
        log.warning(f"⚠️ Ошибка генерации шуток: {e}")
        return None


def load_jokes_from_file():
    """Загрузка шуток из файла."""
    p = Path(__file__).parent / "jokes.txt"
    if not p.exists():
        return ["Тестовая шутка"]
    with open(p, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines if lines else ["Анекдотов нет"]


def get_jokes():
    """
    Получает шутки. Сначала пробует AI, если не вышло — из файла.
    Возвращает список.
    """
    # Пробуем AI
    ai_jokes = generate_jokes_ai(10)
    if ai_jokes:
        return ai_jokes
    
    # Запасной вариант — файл
    file_jokes = load_jokes_from_file()
    log.info(f"📖 Загружено {len(file_jokes)} шуток из файла")
    return file_jokes


def choose_joke(jokes):
    """
    Показывает шутки и просит выбрать.
    Возвращает выбранную шутку.
    """
    print("\n" + "=" * 55)
    print("🎯 Выбери шутку для видео:")
    print("=" * 55)
    
    for i, j in enumerate(jokes, 1):
        print(f"  {i}. {j}")
    
    print(f"  {len(jokes)+1}. 🔄 Сгенерировать новые")
    print(f"  {len(jokes)+2}. 🎲 Случайная")
    print("=" * 55)
    
    while True:
        try:
            choice = input("👉 Номер (1-{}): ".format(len(jokes)+2)).strip()
            if not choice:
                continue
            n = int(choice)
            if 1 <= n <= len(jokes):
                selected = jokes[n-1]
                print(f"✅ Выбрано: {selected}\n")
                return selected
            elif n == len(jokes) + 1:
                # Регенерация
                print("🔄 Генерирую новые шутки...")
                new_jokes = generate_jokes_ai(10)
                if new_jokes:
                    return choose_joke(new_jokes)
                else:
                    print("⚠️ Не удалось сгенерировать, показываю те же")
                    return choose_joke(jokes)
            elif n == len(jokes) + 2:
                choice = random.choice(jokes)
                print(f"🎲 Случайная: {choice}\n")
                return choice
            else:
                print(f"❌ Введи число от 1 до {len(jokes)+2}")
        except ValueError:
            print("❌ Введи число")


# ─── TTS ───

def generate_audio(text, c, mode):
    out = Path(c["temp_dir"]) / "audio.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "light":
        log.info("🎤 Silero TTS...")
        try:
            m = torch.package.PackageImporter(c["tts_path"]).load_pickle("tts_models", "model")
            m.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            a = m.apply_tts(text, speaker=c.get("tts_speaker","xenia"), sample_rate=c.get("tts_sample_rate",24000))
            import soundfile as sf
            sf.write(str(out), a.cpu().numpy(), c.get("tts_sample_rate",24000))
            del m; torch.cuda.empty_cache()
            return str(out)
        except Exception as e:
            log.error(f"Silero: {e}")
    else:
        log.info("🎤 XTTS-v2...")
        try:
            from TTS.api import TTS
            t = TTS(model_path=c["tts_path"], gpu=torch.cuda.is_available())
            t.tts_to_file(text=text, speaker=c.get("tts_speaker","female"), file_path=str(out))
            del t; torch.cuda.empty_cache()
            return str(out)
        except Exception as e:
            log.error(f"XTTS: {e}")
    
    # Заглушка
    import soundfile as sf
    sr = c.get("tts_sample_rate", 24000)
    sf.write(str(out), np.zeros(int(sr * max(3, len(text)/10)), dtype=np.float32), sr)
    return str(out)


# ─── SD ───

def generate_image(prompt, c, mode):
    out = Path(c["temp_dir"]) / "img.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    log.info(f"🎨 Stable Diffusion: {prompt[:40]}...")
    try:
        if mode == "light":
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                c["sd_model"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=True, safety_checker=None, requires_safety_checker=False)
        else:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                c["sd_model"], torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=True)
        
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            pipe.enable_attention_slicing()
        
        img = pipe(
            prompt=f"surreal, funny, cartoon: {prompt}",
            negative_prompt="text, ugly, deformed, blurry",
            num_inference_steps=c.get("sd_step", 25),
            guidance_scale=c.get("sd_guidance_scale", 7.5),
            width=c.get("sd_width", 768), height=c.get("sd_height", 768),
        ).images[0]
        
        img.save(str(out))
        del pipe; torch.cuda.empty_cache()
        return str(out)
    except Exception as e:
        log.error(f"SD: {e}")
        img = Image.new("RGB", (768,768), (random.randint(0,99), random.randint(0,99), random.randint(100,199)))
        ImageDraw.Draw(img).text((50,300), prompt[:50], fill=(255,255,255))
        img.save(str(out))
        return str(out)


# ─── АНИМАЦИЯ ───

def anim_light(img_path, text, dur, c):
    log.info("🎬 Мемная анимация (light)...")
    tw, th, fps = c["video_width"], c["video_height"], c["fps"]
    base = Image.open(img_path).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
    emojis = ["😂","🤣","💀","🔥","😭","🥶","🤡","👀","💯","🗿","😳","🤨"]
    random.seed(datetime.datetime.now().timestamp())
    shake = random.uniform(3, 8)
    
    emoji_data = [
        {"e": random.choice(emojis), "x": random.uniform(0,tw-50), "y": random.uniform(0,th-50),
         "sx": random.uniform(-60,60), "sy": random.uniform(-40,40), "sz": random.randint(30,70), "op": random.uniform(0.3,0.8)}
        for _ in range(random.randint(4,8))
    ]
    flash_times = sorted([random.uniform(0, dur) for _ in range(random.randint(2,4))])
    
    def make_frame(t):
        f = base.copy()
        d = ImageDraw.Draw(f)
        for e in emoji_data:
            x = (e["x"] + e["sx"]*t + math.sin(t*15)*shake) % (tw+50) - 25
            y = (e["y"] + e["sy"]*t + math.cos(t*12)*shake*0.5) % (th+50) - 25
            try:
                d.text((int(x),int(y)), e["e"], font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", e["sz"]),
                       fill=(255,255,255,int(255*e["op"])))
            except: pass
        for t0 in flash_times:
            if abs(t-t0) < 0.1:
                a = max(0, 1-abs(t-t0)*10)
                c = (int(255*a*random.choice([0.8,0.2])), int(255*a*random.choice([0.2,0.8])), int(255*a*random.choice([0.2,0.5])))
                f = Image.blend(f, Image.new("RGB", f.size, c), 0.3*a)
        return np.array(f)
    
    return VideoClip(make_frame, duration=dur).set_fps(fps)


def anim_heavy(img_path, text, dur, c):
    log.info("🎬 AnimateDiff (heavy)...")
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        adapter = MotionAdapter.from_pretrained(c.get("animatediff_motion","models/animatediff-motion"), torch_dtype=torch.float16)
        pipe = AnimateDiffPipeline.from_pretrained(c["sd_model"], motion_adapter=adapter, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_pretrained(c["sd_model"], subfolder="scheduler",
            clip_sample=False, timestep_spacing="linspace", steps_offset=1, beta_schedule="linear", prediction_type="v_prediction")
        pipe.enable_vae_slicing(); pipe.enable_model_cpu_offload()
        
        output = pipe(prompt=f"cinematic, funny: {text[:100]}", negative_prompt="ugly",
                     num_frames=min(24, int(dur*8)), guidance_scale=7.5, num_inference_steps=25)
        frames = output.frames[0]
        tw, th, fps = c["video_width"], c["video_height"], c["fps"]
        
        fd = Path(c["temp_dir"]) / "fr"
        fd.mkdir(parents=True, exist_ok=True)
        clips = []
        for i, frame in enumerate(frames):
            fp = fd / f"f_{i:04d}.png"
            frame.save(str(fp))
            clips.append(ImageClip(str(fp)).resize((tw,th)).set_duration(dur/len(frames)))
        
        result = concatenate_videoclips(clips, method="compose")
        result = result.loop(duration=dur) if result.duration < dur else result.subclip(0, dur)
        del pipe; torch.cuda.empty_cache()
        return result
    except Exception as e:
        log.warning(f"AnimateDiff не сработал: {e}, → light")
        return anim_light(img_path, text, dur, c)


# ─── СУБТИТРЫ ───

def create_subtitles(text, dur, c, mode):
    if not c.get("subtitles_enabled", True):
        return None
    fs = c.get("subtitles_font_size", 36)
    mb = c.get("subtitles_margin_bottom", 80)
    vw, vh = c["video_width"], c["video_height"]
    
    if c.get("subtitles_static", True) or mode == "light":
        try:
            t = TextClip(text, fontsize=fs, color="white", stroke_color="black", stroke_width=2,
                        font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", method="label")
            return t.set_position(("center", vh-mb-fs)).set_duration(dur).crossfadein(0.3)
        except:
            return None
    else:
        words = text.split()
        if not words: return None
        wd = dur / len(words)
        clips = []
        for i in range(len(words)):
            try:
                t = TextClip(" ".join(words[:i+1]), fontsize=fs, color="white", stroke_color="black",
                            stroke_width=2, font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", method="label")
                clips.append(t.set_position(("center", vh-mb-fs)).set_start(i*wd).set_duration(wd*1.5))
            except: pass
        return clips


# ─── СБОРКА ВИДЕО ───

def assemble_video(visual, audio_path, text, c, mode):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(c["output_dir"]) / f"video_{ts}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    log.info("🔧 Сборка видео...")
    a = AudioFileClip(audio_path)
    dur = a.duration
    
    if mode == "light":
        vid = anim_light(visual, text, dur, c)
    else:
        vid = anim_heavy(visual, text, dur, c)
    
    s = create_subtitles(text, dur, c, mode)
    comp = [vid]
    if s is not None:
        comp += s if isinstance(s, list) else [s]
    
    final = CompositeVideoClip(comp, size=(c["video_width"], c["video_height"])) if len(comp) > 1 else vid
    final = final.set_audio(a)
    
    log.info(f"💾 Экспорт в {out}")
    final.write_videofile(str(out), codec="libx264", audio_codec="aac",
                         fps=c.get("fps", 24), preset="medium", bitrate="5000k")
    final.close(); a.close()
    log.info(f"✅ Видео готово: {out}")
    return str(out)


# ─── ПАЙПЛАЙН ───

def run_pipeline(mode):
    print("\n" + "=" * 55)
    print("    🎬  ФАБРИКА ВИДЕО  🎬")
    print("=" * 55)
    
    if torch.cuda.is_available():
        log.info(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB)")
    else:
        log.warning("⚠️ CUDA не найдена! Будет медленно.")
    
    c = load_config(mode)
    
    # 1. Получаем шутки
    jokes = get_jokes()
    
    # 2. Выбираем
    text = choose_joke(jokes)
    
    # 3. Генерируем
    log.info(f"📝 Текст: {text}")
    log.info(f"🎤 Озвучка...")
    a = generate_audio(text, c, mode)
    log.info(f"🎨 Картинка...")
    i = generate_image(text, c, mode)
    log.info(f"🎞️ Видео...")
    assemble_video(i, a, text, c, mode)
    
    # Спрашиваем, хочет ли ещё
    try:
        again = input("\n🔄 Ещё одно видео? (Enter — да, n — нет): ").strip().lower()
        if again != "n":
            run_pipeline(mode)
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Фабрика видео")
    parser.add_argument("--mode", choices=["light","heavy","ultra"], default="light")
    args = parser.parse_args()
    try:
        run_pipeline(args.mode)
    except KeyboardInterrupt:
        log.info("⏹️ Пока!"); sys.exit(0)
    except Exception as e:
        log.exception(f"❌ {e}"); sys.exit(1)
