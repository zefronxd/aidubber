"""AI Video Dubber: bade videos (1-2 GB) ke liye. Chunked upload, disk-backed audio, streaming ffmpeg."""
import asyncio, hmac, io, os, re, shutil, subprocess, threading, time, uuid, wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse

SR, CHUNK = 32000, 8 << 20
MAX_MB = int(os.getenv("MAX_MB", "2500"))
MAX_MIN = int(os.getenv("MAX_MIN", "180"))
LOCAL_MAX_MIN = int(os.getenv("LOCAL_MAX_MIN", "15"))   # bina cloud key ke local Whisper ki limit (VPS par badha sakte ho)
ACCESS_CODE = os.getenv("ACCESS_CODE", "")   # set karo to app par password lagta hai
ROOT = Path(__file__).parent
WORK = ROOT / "work"; WORK.mkdir(exist_ok=True)
FFMPEG = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
VOICES = {"hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
          "en": {"female": "en-IN-NeerjaNeural", "male": "en-IN-PrabhatNeural"}}
jobs, pool = {}, ThreadPoolExecutor(max_workers=1)
_model, _lock = None, threading.Lock()
app = FastAPI()

def guard(request: Request):  # ACCESS_CODE set ho to cookie chahiye (Groq key/disk ka dusre log use na karein)
    if ACCESS_CODE and not hmac.compare_digest(request.cookies.get("code", ""), ACCESS_CODE):
        raise HTTPException(401, "Access code chahiye.")

@app.post("/api/login")
def login(response: Response, code: str = Form("")):
    if not ACCESS_CODE or hmac.compare_digest(code, ACCESS_CODE):
        response.set_cookie("code", ACCESS_CODE, httponly=True, samesite="lax", max_age=86400 * 30)
        return {"ok": True}
    raise HTTPException(401, "Galat code.")

def has_cloud(): return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))

def ff(*a, data=None):
    return subprocess.run([FFMPEG, "-v", "error", "-y", *a], input=data, capture_output=True, check=True).stdout

def pcm(data, atempo=None):
    a = ["-i", "pipe:0"] + (["-filter:a", f"atempo={atempo:.3f}"] if atempo else [])
    return np.frombuffer(ff(*a, "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1", data=data), dtype=np.int16)

def whisper():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), compute_type=os.getenv("WHISPER_COMPUTE", "int8"))
    return _model

def cloud_stt(url, key, model, wav, d, st):
    for f in d.glob("seg*.mp3"): f.unlink()
    ff("-i", str(wav), "-f", "segment", "-segment_time", "900", "-c:a", "libmp3lame", "-b:a", "32k", str(d / "seg%03d.mp3"))
    parts, out, lang = sorted(d.glob("seg*.mp3")), [], ""
    for k, p in enumerate(parts):          # 15 minute ke tukde: bade video bhi chalte hain
        for t in range(4):
            with open(p, "rb") as f:
                r = requests.post(url, headers={"Authorization": "Bearer " + key}, files={"file": ("a.mp3", f, "audio/mpeg")},
                                  data={"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "segment"}, timeout=300)
            if r.status_code == 200: break
            if r.status_code != 429 or t == 3: raise RuntimeError("Cloud STT error: " + r.text[:150])
            time.sleep(20 * (t + 1))       # rate limit: ruk ke dobara
        j, off = r.json(), k * 900
        lang = lang or j.get("language", "")
        out += [{"s": g["start"] + off, "e": g["end"] + off, "o": g["text"].strip()} for g in j.get("segments", [])
                if g["text"].strip() and g.get("no_speech_prob", 0) < 0.85]
        st("Awaaz ko text bana raha hoon", 12 + int(33 * (k + 1) / len(parts)))
    return out, lang

def local_stt(wav, st):
    try: it, info = whisper().transcribe(str(wav), vad_filter=True, beam_size=int(os.getenv("WHISPER_BEAM", "1")))
    except ImportError: raise RuntimeError("faster-whisper install nahi hai.")
    raw = []
    for s in it:
        if s.text.strip(): raw.append({"s": s.start, "e": s.end, "o": s.text.strip()})
        st("Awaaz ko text bana raha hoon", 12 + int(33 * min(s.end / max(info.duration, 1), 1)))
    return raw, info.language

def stt(wav, d, st, dur):
    """Groq/OpenAI Whisper (key ho to), fail ya key na ho to local faster-whisper."""
    mins, cloud = dur / 60, None
    if os.getenv("GROQ_API_KEY"):
        cloud = ("https://api.groq.com/openai/v1/audio/transcriptions", os.environ["GROQ_API_KEY"], os.getenv("STT_MODEL", "whisper-large-v3"))
    elif os.getenv("OPENAI_API_KEY"):
        cloud = ("https://api.openai.com/v1/audio/transcriptions", os.environ["OPENAI_API_KEY"], "whisper-1")
    if mins > MAX_MIN: raise RuntimeError(f"Video {MAX_MIN} minute se chhota hona chahiye (abhi {mins:.0f} min).")
    err = ""
    if cloud:
        try: return cloud_stt(*cloud, wav, d, st)
        except Exception as e: err = str(e); st("Cloud fail hua, local Whisper chal raha hai", 12)
    if mins > LOCAL_MAX_MIN: raise RuntimeError((err + " | " if err else "") + f"{LOCAL_MAX_MIN} min se bade video ke liye GROQ_API_KEY zaroori hai.")
    try: return local_stt(wav, st)
    except RuntimeError as e: raise RuntimeError((err + " | " if err else "") + str(e))

def merge(segs):  # tootay fragments ko poore sentence banao
    out = []
    for s in segs:
        p = out[-1] if out else None
        if p and s["s"] - p["e"] < 0.6 and s["e"] - p["s"] < 14 and (not re.search(r"[.!?。！？]$", p["o"]) or p["e"] - p["s"] < 1.5):
            p["e"], p["o"] = s["e"], p["o"] + " " + s["o"]
        else:
            out.append(dict(s))
    return out

def translate(texts, lang):  # Google Translate: ~4000 char ke batch mein (kam requests, rate-limit se bachav)
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source="auto", target=lang)
    def call(t):
        for i in range(4):
            try: return tr.translate(t) or t
            except Exception: time.sleep(1.5 * (i + 1))
        return None
    out, i = [None] * len(texts), 0
    while i < len(texts):
        j, n = i, 0
        while j < len(texts) and n + len(texts[j]) < 4000: n += len(texts[j]) + 1; j += 1
        j = max(j, i + 1)
        r = call("\n".join(t.replace("\n", " ") for t in texts[i:j]))
        parts = r.split("\n") if r else []
        if len(parts) == j - i: out[i:j] = parts
        else:
            for k in range(i, j): out[k] = call(texts[k]) or texts[k]
        i = j
    return out

_edge_bad_until = 0

def gtts_bytes(text, lang):
    from gtts import gTTS
    b = io.BytesIO(); gTTS(text, lang=lang).write_to_fp(b); return b.getvalue()

async def tts(text, voice, rate="+0%"):
    global _edge_bad_until
    import edge_tts
    for i in range(3 if time.time() > _edge_bad_until else 0):
        try:
            buf = b""
            async for c in edge_tts.Communicate(text, voice, rate=rate).stream():
                if c["type"] == "audio": buf += c["data"]
            if buf: return buf
        except Exception:
            await asyncio.sleep(1 + i)
    _edge_bad_until = time.time() + 300          # Edge TTS fail: 5 min Google TTS par raho
    try: return await asyncio.to_thread(gtts_bytes, text, voice[:2])
    except Exception: raise RuntimeError("Awaaz (TTS) nahi ban payi. Internet check karo.")

async def speak(text, voice, slot):  # time slot mein fit: pehle TTS speed, phir atempo
    mp3 = await tts(text, voice)
    a = await asyncio.to_thread(pcm, mp3)
    if len(a) / SR > slot * 1.03:
        mp3 = await tts(text, voice, f"+{min(int((len(a) / SR / slot - 1) * 100) + 4, 40)}%")
        a = await asyncio.to_thread(pcm, mp3)
        d = len(a) / SR
        if d > slot * 1.03: a = await asyncio.to_thread(pcm, mp3, min(d / slot, 1.3))
    return a

async def tts_batch(segs, idx, voice):
    sem = asyncio.Semaphore(4)
    async def one(i):
        async with sem:
            nxt = segs[i + 1]["s"] if i + 1 < len(segs) else segs[i]["e"] + 3
            return await speak(segs[i]["t"], voice, max(nxt - segs[i]["s"] - 0.08, 0.8))
    return await asyncio.gather(*(one(i) for i in idx))

def write_srt(segs, path):
    def ts(x): return f"{int(x // 3600):02}:{int(x % 3600 // 60):02}:{int(x % 60):02},{int(x % 1 * 1000):03}"
    with open(path, "w", encoding="utf-8") as f:
        for n, g in enumerate(segs, 1): f.write(f"{n}\n{ts(g['s'])} --> {ts(g['e'])}\n{g['t']}\n\n")

def process(j, url):
    def st(s, p): j.update(stage=s, pct=p)
    d, src, lang, gender, keep = j["dir"], j.get("src"), j["lang"], j["gender"], j["keep"]
    wav, dub = d / "a16.wav", None
    try:
        if url:
            st("Video download ho raha hai", 3)
            subprocess.run(["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]", "--merge-output-format", "mp4", "--ffmpeg-location", FFMPEG,
                            "-o", str(d / "in.%(ext)s"), url], check=True, capture_output=True)
            src = next(d.glob("in.*"))
        st("Audio nikaal raha hoon", 8)
        ff("-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav))
        with wave.open(str(wav)) as w: dur = w.getnframes() / 16000
        st("Awaaz ko text bana raha hoon", 12)
        raw, src_lang = stt(wav, d, st, dur)
        if not raw: raise RuntimeError("Video mein bolti hui awaaz nahi mili.")
        segs = merge(raw)
        st("Google Translate se translate ho raha hai", 48)
        for g, t in zip(segs, translate([g["o"] for g in segs], lang)): g["t"] = t
        voice, total, cur = VOICES[lang][gender], int((dur + 40) * SR), 0
        dub = np.memmap(d / "dub.raw", dtype=np.int16, mode="w+", shape=(total,))   # RAM nahi, disk par
        for b in range(0, len(segs), 40):
            idx = list(range(b, min(b + 40, len(segs))))
            for i, a in zip(idx, asyncio.run(tts_batch(segs, idx, voice))):
                p = max(int(segs[i]["s"] * SR), cur); n = max(0, min(len(a), total - p))
                dub[p:p + n] = a[:n]; cur = p + n + int(0.05 * SR)
            st("Dubbing awaaz ban rahi hai", 55 + int(35 * (idx[-1] + 1) / len(segs)))
        dub.flush(); del dub; dub = None
        tdur = min(max(dur, cur / SR), total / SR)
        st("Audio taiyaar ho raha hai", 92)
        m4a, rawin = d / "dubbed.m4a", ["-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", str(d / "dub.raw")]
        enc = ["-t", f"{tdur:.2f}", "-c:a", "aac", "-b:a", "160k", str(m4a)]
        try:
            if not keep: raise subprocess.CalledProcessError(1, "x")
            ff(*rawin, "-i", str(src), "-filter_complex",
               "[1:a:0]volume=0.15[o];[0:a][o]amix=inputs=2:duration=first:dropout_transition=0,volume=2[m]", "-map", "[m]", *enc)
        except subprocess.CalledProcessError:
            ff(*rawin, *enc)
        write_srt(segs, d / "dubbed.srt")
        for p in [wav, d / "dub.raw", *d.glob("seg*.mp3")]: p.unlink(missing_ok=True)   # mux se pehle disk khali
        vid = None
        try:
            st("Video mein audio laga raha hoon", 95)
            if shutil.disk_usage(d).free < src.stat().st_size * 1.05 + (50 << 20): raise OSError("disk kam")
            fast = ["-movflags", "+faststart"] if src.stat().st_size < 500 << 20 else []
            ff("-i", str(src), "-i", str(m4a), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", *fast, "-shortest", str(d / "dubbed.mp4"))
            vid = f"/api/files/{d.name}/dubbed.mp4"
        except Exception:
            (d / "dubbed.mp4").unlink(missing_ok=True)   # disk kam / format issue: audio to mil hi jayegi
        j.update(status="done", stage="Ho gaya", pct=100, result={
            "video": vid, "audio": f"/api/files/{d.name}/dubbed.m4a", "srt": f"/api/files/{d.name}/dubbed.srt", "lang": src_lang,
            "segments": [{"s": round(g["s"], 1), "o": g["o"], "t": g["t"]} for g in segs]})
    except Exception as e:
        msg = e.stderr.decode(errors="ignore")[-300:] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        if url and not src: msg = "Link se video nahi aaya (YouTube block kar sakta hai). Video file upload karo. " + msg
        j.update(status="error", error=msg or repr(e))
    finally:
        del dub
        for p in [src, wav, d / "dub.raw", *d.glob("seg*.mp3")]:   # disk khali karo
            if p: Path(p).unlink(missing_ok=True)

def new_job(lang, voice, keep):
    if lang not in VOICES or voice not in ("female", "male"): raise HTTPException(400, "Galat option.")
    for k, v in list(jobs.items()):
        if time.time() - v["t"] > 10800: shutil.rmtree(v["dir"], True); jobs.pop(k)
    jid = uuid.uuid4().hex[:12]; d = WORK / jid; d.mkdir()
    j = jobs[jid] = {"t": time.time(), "dir": d, "status": "up", "stage": "Upload", "pct": 1,
                     "lang": lang, "gender": voice, "keep": keep == "1", "next": 0, "got": 0, "lock": threading.Lock()}
    return jid, j

@app.post("/api/jobs/init", dependencies=[Depends(guard)])
async def init(name: str = Form(""), size: int = Form(0), url: str = Form(""), lang: str = Form("hi"),
               voice: str = Form("female"), keep: str = Form("0")):
    if name:
        if size <= 0 or size > MAX_MB << 20: raise HTTPException(413, f"Video {MAX_MB}MB se chhota hona chahiye.")
        free = shutil.disk_usage(WORK).free
        if free < size + (800 << 20): raise HTTPException(507, f"Server par disk jagah kam hai ({free >> 20} MB free). Chhota video do.")
    elif not url.startswith("http"): raise HTTPException(400, "Video upload karo ya link do.")
    jid, j = new_job(lang, voice, keep)
    if name:
        j.update(size=size, src=j["dir"] / ("in" + re.sub(r"[^.\w]", "", Path(name).suffix.lower())[:8]))
    else:
        j.update(status="run", stage="Line mein hoon"); pool.submit(process, j, url)
    return {"id": jid, "chunk": CHUNK}

@app.post("/api/jobs/{jid}/chunk", dependencies=[Depends(guard)])
async def chunk(jid: str, i: int, request: Request):
    j = jobs.get(jid)
    if not j or j["status"] != "up": raise HTTPException(404, "Upload session nahi mila.")
    data = await request.body()
    with j["lock"]:
        if i > j["next"]: raise HTTPException(409, "Chunk order galat.")
        if i == j["next"]:                      # retry par duplicate chunk ignore hota hai
            if j["got"] + len(data) > j["size"]: raise HTTPException(413, "File size mismatch.")
            with open(j["src"], "ab") as f: f.write(data)
            j["got"] += len(data); j["next"] += 1
    return {"next": j["next"]}

@app.post("/api/jobs/{jid}/start", dependencies=[Depends(guard)])
def start(jid: str):
    j = jobs.get(jid)
    if not j or j["status"] != "up": raise HTTPException(404, "Upload session nahi mila.")
    if j["got"] != j["size"]: raise HTTPException(400, "Upload poora nahi hua.")
    j.update(status="run", stage="Line mein hoon", pct=1)
    pool.submit(process, j, "")
    return {"ok": True}

@app.get("/api/jobs/{jid}", dependencies=[Depends(guard)])
def status(jid: str):
    j = jobs.get(jid)
    if not j: raise HTTPException(404, "Job nahi mila.")
    return {k: j.get(k) for k in ("status", "stage", "pct", "error", "result", "next", "got", "size")}

@app.get("/api/files/{jid}/{name}", dependencies=[Depends(guard)])
def get_file(jid: str, name: str, dl: int = 0):
    j = jobs.get(jid)
    if name not in ("dubbed.mp4", "dubbed.m4a", "dubbed.srt") or not j or not (j["dir"] / name).exists(): raise HTTPException(404, "File nahi mili.")
    return FileResponse(j["dir"] / name, media_type={"mp4": "video/mp4", "m4a": "audio/mp4", "srt": "text/plain; charset=utf-8"}[name[-3:]], filename=name if dl else None)

@app.get("/api/config")
def config(): return {"max_mb": MAX_MB, "max_min": MAX_MIN if has_cloud() else LOCAL_MAX_MIN}

@app.on_event("startup")
def warm():  # bina key ke: Whisper model background mein load
    if not has_cloud():
        def load():
            try: whisper()
            except Exception: pass
        threading.Thread(target=load, daemon=True).start()

@app.get("/")
def index(): return FileResponse(ROOT / "index.html")
