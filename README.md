# AI Video Dubber (Heroku)

## Deploy
    git init && git add . && git commit -m init
    heroku login
    heroku create apna-app-name
    # optional: heroku config:set GROQ_API_KEY=gsk_xxxxxxxx
    git push heroku HEAD:main        # branch master ho to: git push heroku master
    heroku ps:scale web=1
    heroku open

Logs: heroku logs --tail

## Zaroori baatein
- Sirf 1 web dyno rakho (jobs memory mein hain). Dyno restart hone par purane jobs/videos delete ho jate hain.
- Basic/Eco dyno (512MB RAM): chhote videos (<=15 min, <=100MB) rakho. MAX_MIN / MAX_MB se badlo.
- Speech-to-text: bina key ke bhi chalta hai (dyno par faster-whisper, default model `base`).
  Behtar/tez quality ke liye GROQ_API_KEY (console.groq.com) ya OPENAI_API_KEY set karo, tab cloud Whisper use hoga.
- Bina key mode: pehli request se pehle model download hota hai (dyno restart par dobara). `heroku config:set WHISPER_MODEL=tiny|base|small`.
  RAM kam pade (R14 error) to chhota model / chhota video lo, ya `heroku ps:resize web=standard-2x`.
- ffmpeg imageio-ffmpeg se aata hai, buildpack nahi chahiye.
- Link wale videos (yt-dlp) Heroku IP par aksar YouTube block kar deta hai. Video file upload karo.
- Local run: pip install -r requirements.txt && uvicorn app:app --port 8000

## Bade video (1-2 GB)
- Video 8MB ke tukdon mein upload hota hai (Heroku timeout nahi lagta), server par sirf audio process hoti hai, video copy hota hai (re-encode nahi).
- Bade/lambe video ke liye GROQ_API_KEY zaroori hai. Bina key ke sirf 15 min tak.
- Heroku disk temporary hai: 2GB video + dubbed output ek saath disk par rehte hain. Jagah kam pade to dubbed.mp4 nahi banegi, dubbed.m4a (audio) mil jayegi:
  ffmpeg -i original.mp4 -i dubbed.m4a -map 0:v -map 1:a -c copy -shortest dubbed.mp4

## Extra features
- ACCESS_CODE: `heroku config:set ACCESS_CODE=kuchbhi` -> app par password lag jata hai (warna koi bhi tumhari Groq quota/disk use kar sakta hai).
- Upload beech mein toot jaye to wahi file dobara chuno, upload wahin se aage badhega. Page refresh karne par bhi job dikhta rahega.
- Output: dubbed.mp4 + dubbed.m4a (audio) + dubbed.srt (translated subtitles).
- Edge TTS fail ho to Google TTS (gTTS) par khud switch ho jata hai.
- VPS/Docker: `docker build -t dubber . && docker run -p 8000:8000 -e GROQ_API_KEY=... dubber`
  Bina Groq key ke VPS par bada video: `-e LOCAL_MAX_MIN=120 -e WHISPER_MODEL=small` (CPU par dheema hoga).
