#!/usr/bin/env python3
"""Фабрика видео. Запуск: python start.py"""

import os, sys, re, random, argparse, logging, datetime, math
from pathlib import Path
import yaml, torch, numpy as np
from PIL import Image, ImageDraw
from moviepy.editor import VideoClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip, concatenate_videoclips, VideoFileClip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def cfg(mode):
    p = Path(__file__).parent/f"config_{mode}.yaml"
    if not p.exists(): log.error(f"Нет {p}"); sys.exit(1)
    c = yaml.safe_load(open(p))
    base = Path(__file__).parent
    for k in ("models_dir","output_dir","temp_dir"):
        if c.get(k) and not Path(c[k]).is_absolute(): c[k]=str(base/c[k])
    for k in ("sd_model","tts_path","face_image","wav2lip_path","gfpgan_path","realesrgan_path",
              "ip_adapter_path","animatediff_motion","animatediff_lightning"):
        if c.get(k) and not Path(c[k]).is_absolute(): c[k]=str(base/c[k])
    log.info(f"Конфиг: {p.name}"); return c

def joke():
    p = Path(__file__).parent/"jokes.txt"
    if not p.exists(): return "Тест"
    l = [l.strip() for l in open(p,encoding="utf-8") if l.strip()]
    return random.choice(l) if l else "Нет анекдотов"

def audio(text, c, mode):
    out = Path(c["temp_dir"])/"audio.wav"; out.parent.mkdir(parents=True,exist_ok=True)
    if mode=="light":
        log.info("🎤 Silero...")
        try:
            m = torch.package.PackageImporter(c["tts_path"]).load_pickle("tts_models","model")
            m.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            a = m.apply_tts(text,speaker=c.get("tts_speaker","xenia"),sample_rate=c.get("tts_sample_rate",24000))
            import soundfile as sf; sf.write(str(out),a.cpu().numpy(),c.get("tts_sample_rate",24000))
            del m; torch.cuda.empty_cache(); return str(out)
        except Exception as e: log.error(f"Silero: {e}")
    else:
        log.info("🎤 XTTS...")
        try:
            from TTS.api import TTS
            t = TTS(model_path=c["tts_path"],gpu=torch.cuda.is_available())
            t.tts_to_file(text=text, speaker=c.get("tts_speaker","female"), file_path=str(out))
            del t; torch.cuda.empty_cache(); return str(out)
        except Exception as e: log.error(f"XTTS: {e}")
    import soundfile as sf
    sr=c.get("tts_sample_rate",24000); sf.write(str(out),np.zeros(int(sr*max(3,len(text)/10)),dtype=np.float32),sr)
    return str(out)

def image(prompt, c, mode):
    out = Path(c["temp_dir"])/"img.png"; out.parent.mkdir(parents=True,exist_ok=True)
    log.info(f"🎨 SD: {prompt[:40]}...")
    try:
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        d = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if torch.cuda.is_available() else torch.float32
        if mode=="light":
            pipe = StableDiffusionPipeline.from_pretrained(c["sd_model"],torch_dtype=dt,local_files_only=True,safety_checker=None,requires_safety_checker=False)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(c["sd_model"],torch_dtype=dt,local_files_only=True)
        pipe = pipe.to(d)
        if d=="cuda": pipe.enable_attention_slicing()
        img = pipe(prompt=f"surreal, funny: {prompt}", negative_prompt="text, ugly",
                   num_inference_steps=c.get("sd_step",25), guidance_scale=c.get("sd_guidance_scale",7.5),
                   width=c.get("sd_width",768), height=c.get("sd_height",768)).images[0]
        img.save(str(out)); del pipe; torch.cuda.empty_cache(); return str(out)
    except Exception as e:
        log.error(f"SD: {e}")
        img=Image.new("RGB",(768,768),(random.randint(0,99),random.randint(0,99),random.randint(100,199)))
        ImageDraw.Draw(img).text((50,300),prompt[:50],fill=(255,255,255)); img.save(str(out)); return str(out)

def anim_light(img_path, text, dur, c):
    log.info("🎬 Light анимация...")
    tw,th,fps=c["video_width"],c["video_height"],c["fps"]
    base=Image.open(img_path).convert("RGB").resize((tw,th),Image.Resampling.LANCZOS)
    emojis=["😂","🤣","💀","🔥","😭","🥶","🤡","👀","💯","🗿"]
    random.seed(datetime.datetime.now().timestamp())
    sh=random.uniform(3,8)
    ed=[{"e":random.choice(emojis),"x":random.uniform(0,tw-50),"y":random.uniform(0,th-50),
         "sx":random.uniform(-60,60),"sy":random.uniform(-40,40),"sz":random.randint(30,70),"op":random.uniform(0.3,0.8)}
        for _ in range(random.randint(4,8))]
    ft=sorted([random.uniform(0,dur) for _ in range(random.randint(2,4))])
    def frm(t):
        f=base.copy(); d=ImageDraw.Draw(f)
        for e in ed:
            x=(e["x"]+e["sx"]*t+math.sin(t*15)*sh)%(tw+50)-25
            y=(e["y"]+e["sy"]*t+math.cos(t*12)*sh*0.5)%(th+50)-25
            try: d.text((int(x),int(y)),e["e"],font=ImageFont.truetype(find_font() or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",e["sz"]),fill=(255,255,255,int(255*e["op"])))
            except: pass
        for t0 in ft:
            if abs(t-t0)<0.1:
                a=max(0,1-abs(t-t0)*10)
                ov=Image.new("RGB",f.size,(int(255*a*random.choice([0.8,0.2])),int(255*a*random.choice([0.2,0.8])),int(255*a*random.choice([0.2,0.5]))))
                f=Image.blend(f,ov,0.3*a)
        return np.array(f)
    return VideoClip(frm,duration=dur).set_fps(fps)

def anim_heavy(img_path, text, dur, c):
    log.info("🎬 AnimateDiff...")
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        a=MotionAdapter.from_pretrained(c.get("animatediff_motion","models/animatediff-motion"),torch_dtype=torch.float16)
        pipe=AnimateDiffPipeline.from_pretrained(c["sd_model"],motion_adapter=a,torch_dtype=torch.float16)
        pipe.scheduler=DDIMScheduler.from_pretrained(c["sd_model"],subfolder="scheduler",clip_sample=False,timestep_spacing="linspace",steps_offset=1,beta_schedule="linear",prediction_type="v_prediction")
        pipe.enable_vae_slicing(); pipe.enable_model_cpu_offload()
        out=pipe(prompt=f"cinematic, funny, {text[:100]}",negative_prompt="low quality",num_frames=min(24,int(dur*8)),guidance_scale=7.5,num_inference_steps=25)
        frames=out.frames[0]; tw,th=c["video_width"],c["video_height"]; fps=c["fps"]
        fd=Path(c["temp_dir"])/"fr"; fd.mkdir(parents=True,exist_ok=True)
        clips=[ImageClip(str(fd/f"f_{i:04d}.png")).resize((tw,th)).set_duration(dur/len(frames))
               for i,fr in enumerate(frames) if (fr.save(str(fd/f"f_{i:04d}.png")) or True)]
        r=concatenate_videoclips(clips,method="compose")
        r=r.loop(duration=dur) if r.duration<dur else r.subclip(0,dur)
        del pipe; torch.cuda.empty_cache(); return r
    except Exception as e:
        log.warning(f"AnimateDiff: {e}, fallback light"); return anim_light(img_path,text,dur,c)

def subs(text, dur, c, mode):
    if not c.get("subtitles_enabled",True): return None
    fs=c.get("subtitles_font_size",36); mb=c.get("subtitles_margin_bottom",80); vw,vh=c["video_width"],c["video_height"]
    if c.get("subtitles_static",True) or mode=="light":
        try:
            t=TextClip(text,fontsize=fs,color="white",stroke_color="black",stroke_width=2,font=find_font(),method="label")
            return t.set_position(("center",vh-mb-fs)).set_duration(dur).crossfadein(0.3)
        except: return None
    else:
        words=text.split()
        if not words: return None
        wd=dur/len(words); cls=[]
        for i in range(len(words)):
            try:
                t=TextClip(" ".join(words[:i+1]),fontsize=fs,color="white",stroke_color="black",stroke_width=2,font=find_font(),method="label")
                cls.append(t.set_position(("center",vh-mb-fs)).set_start(i*wd).set_duration(wd*1.5))
            except: pass
        return cls

def find_font():
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","C:/Windows/Fonts/arial.ttf","C:/Windows/Fonts/calibri.ttf"]:
        if os.path.exists(f): return f
    return None

def assemble(vis, aud, text, c, mode):
    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(c["output_dir"])/f"video_{ts}.mp4"; out.parent.mkdir(parents=True,exist_ok=True)
    a=AudioFileClip(aud); dur=a.duration
    if mode=="light": vid=anim_light(vis,text,dur,c)
    elif mode=="heavy": vid=anim_heavy(vis,text,dur,c)
    else:
        wl=c.get("wav2lip_path","models/wav2lip"); fc=c.get("face_image","models/face.png")
        if os.path.exists(wl) and os.path.exists(fc):
            try:
                sys.path.insert(0,os.path.join(wl,"Wav2Lip"))
                from inference import main as wl_main
            except: pass
        vid=anim_heavy(vis,text,dur,c)
    s=subs(text,dur,c,mode); comp=[vid]
    if s is not None: comp+=s if isinstance(s,list) else [s]
    f=CompositeVideoClip(comp,size=(c["video_width"],c["video_height"])) if len(comp)>1 else vid
    f=f.set_audio(a)
    log.info(f"💾 Экспорт: {out}")
    f.write_videofile(str(out),codec="libx264",audio_codec="aac",fps=c.get("fps",24),preset="medium",bitrate="5000k")
    f.close(); a.close(); log.info(f"✅ {out}"); return str(out)

def run_pipeline(mode):
    log.info(f"🚀 Фабрика видео — {mode}")
    if torch.cuda.is_available():
        log.info(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB)")
    else: log.warning("⚠️ CPU — медленно")
    c=cfg(mode); j=joke(); log.info(f"📝 {j}")
    a=audio(j,c,mode); i=image(j,c,mode); assemble(i,a,j,c,mode)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["light","heavy","ultra"],default="light")
    a=p.parse_args()
    try: run_pipeline(a.mode)
    except KeyboardInterrupt: log.info("⏹️"); sys.exit(0)
    except Exception as e: log.exception(f"❌ {e}"); sys.exit(1)
