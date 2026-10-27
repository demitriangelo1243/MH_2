import os, io, re, threading
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from kokoro import KPipeline
from pedalboard import (
    Pedalboard, Reverb, Delay, Chorus, Distortion, Compressor, Limiter,
    HighpassFilter, LowpassFilter, LowShelfFilter, HighShelfFilter,
    PeakFilter, Phaser, Bitcrush, PitchShift, Gain, Clipping
)
from pedalboard.io import AudioFile

SR = 24000
MAX_CHARS = int(os.getenv("MAX_CHARS", "400"))
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://moneyhamster.shop").split(",") if o.strip()]

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ORIGINS}}, methods=["GET", "POST", "OPTIONS"])

# ---------- Kokoro ----------
VOICES = {
    "af_heart": "Heart (US F)", "af_bella": "Bella (US F)", "af_nicole": "Nicole (US F, soft)",
    "af_sarah": "Sarah (US F)", "af_sky": "Sky (US F)", "af_nova": "Nova (US F)",
    "am_adam": "Adam (US M)", "am_michael": "Michael (US M)", "am_fenrir": "Fenrir (US M)",
    "am_puck": "Puck (US M)", "am_onyx": "Onyx (US M, deep)", "am_echo": "Echo (US M)",
    "bf_emma": "Emma (UK F)", "bf_isabella": "Isabella (UK F)", "bf_alice": "Alice (UK F)",
    "bm_george": "George (UK M)", "bm_lewis": "Lewis (UK M)", "bm_daniel": "Daniel (UK M)",
    "bm_fable": "Fable (UK M)",
}
_pipes, _lock = {}, threading.Lock()

def pipe_for(voice):
    lang = voice[0]  # 'a' American, 'b' British
    with _lock:
        if lang not in _pipes:
            _pipes[lang] = KPipeline(lang_code=lang)
        return _pipes[lang]

def resolve_voice(pipe, v1, v2, blend):
    """Blend two voices for a unique custom voice (Kokoro voice tensors)."""
    if v2 and v2 != v1 and blend > 0:
        t1, t2 = pipe.load_voice(v1), pipe.load_voice(v2)
        return t1 * (1 - blend) + t2 * blend
    return v1

def synth(text, voice, voice2, blend, speed):
    p = pipe_for(voice)
    v = resolve_voice(p, voice, voice2 if voice2 and voice2[0] == voice[0] else None, blend)
    parts = []
    for _, _, audio in p(text, voice=v, speed=speed, split_pattern=r"\n+"):
        parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        parts.append(np.zeros(int(SR * 0.12), dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)

# ---------- Effect presets ----------
# Each builder takes m (0..1 intensity) and returns a list of plugins.
def P(label, group, fn): return {"label": label, "group": group, "fn": fn}

PRESETS = {
    # Clean / broadcast
    "clean": P("Clean (no FX)", "Broadcast", lambda m: []),
    "radio_fm": P("FM Radio Voice", "Broadcast", lambda m: [
        HighpassFilter(90), LowShelfFilter(200, gain_db=4*m+1), PeakFilter(3500, gain_db=4*m+1, q=1),
        HighShelfFilter(9000, gain_db=3*m), Compressor(-22, 4+2*m, 5, 80), Limiter(-1)]),
    "deep_announcer": P("Deep Announcer", "Broadcast", lambda m: [
        PitchShift(-2*m-1), LowShelfFilter(150, gain_db=6*m), Compressor(-24, 5, 4, 90),
        Reverb(room_size=0.15, wet_level=0.08, dry_level=0.9), Limiter(-1)]),
    "bright_presenter": P("Bright Presenter", "Broadcast", lambda m: [
        HighpassFilter(110), PeakFilter(5000, gain_db=5*m, q=0.9), HighShelfFilter(10000, gain_db=4*m),
        Compressor(-20, 3.5, 3, 70), Limiter(-1)]),
    "podcast_warm": P("Warm Podcast", "Broadcast", lambda m: [
        HighpassFilter(70), LowShelfFilter(180, gain_db=3*m), PeakFilter(2500, gain_db=-2*m),
        Compressor(-20, 3, 8, 120), Limiter(-1)]),
    "heavy_comp": P("Heavy Compression", "Broadcast", lambda m: [
        Compressor(-30, 8+4*m, 2, 60), Gain(6*m), Limiter(-0.5)]),
    # Space
    "club_hall": P("Club Hall", "Space", lambda m: [
        Reverb(room_size=0.5+0.3*m, damping=0.5, wet_level=0.25*m+0.1, dry_level=0.8, width=1.0)]),
    "stadium": P("Stadium", "Space", lambda m: [
        Reverb(room_size=0.95, damping=0.3, wet_level=0.35*m+0.1, dry_level=0.7),
        Delay(delay_seconds=0.25, feedback=0.25, mix=0.2*m)]),
    "cathedral": P("Cathedral", "Space", lambda m: [
        Reverb(room_size=1.0, damping=0.2, wet_level=0.5*m+0.1, dry_level=0.6)]),
    "tight_booth": P("Tight Booth", "Space", lambda m: [
        Reverb(room_size=0.08, damping=0.7, wet_level=0.1*m+0.03, dry_level=0.95)]),
    "tunnel": P("Tunnel", "Space", lambda m: [
        Reverb(room_size=0.85, damping=0.1, wet_level=0.4*m+0.1, dry_level=0.6),
        LowpassFilter(6000)]),
    # Echo / rhythm
    "slapback": P("Slapback Echo", "Echo", lambda m: [Delay(delay_seconds=0.12, feedback=0.1, mix=0.35*m+0.1)]),
    "dub_echo": P("Dub Echo", "Echo", lambda m: [
        Delay(delay_seconds=0.375, feedback=0.5*m+0.2, mix=0.4*m+0.1), LowpassFilter(5000),
        Reverb(room_size=0.4, wet_level=0.15)]),
    "ping_pong_128": P("Tempo Echo 128 BPM", "Echo", lambda m: [
        Delay(delay_seconds=60/128*0.75, feedback=0.4*m+0.15, mix=0.35*m+0.1)]),
    "tape_echo": P("Tape Echo", "Echo", lambda m: [
        Delay(delay_seconds=0.3, feedback=0.45*m, mix=0.3), LowpassFilter(3500),
        Chorus(rate_hz=0.4, depth=0.3, mix=0.3), Distortion(4*m)]),
    # Character / pitch
    "chipmunk": P("Chipmunk", "Pitch", lambda m: [PitchShift(4+4*m)]),
    "giant": P("Giant", "Pitch", lambda m: [PitchShift(-4-4*m), Reverb(room_size=0.4, wet_level=0.15)]),
    "robot": P("Robot", "Pitch", lambda m: [
        Chorus(rate_hz=60, depth=0.9, centre_delay_ms=1.5, feedback=0.5, mix=0.6*m+0.3),
        Bitcrush(bit_depth=12), Limiter(-1)]),
    "alien": P("Alien", "Pitch", lambda m: [
        PitchShift(3), Phaser(rate_hz=1.5, depth=0.8, mix=0.6), Reverb(room_size=0.5, wet_level=0.2)]),
    "demon": P("Demon", "Pitch", lambda m: [
        PitchShift(-6-3*m), Distortion(10+10*m), LowpassFilter(5000),
        Reverb(room_size=0.7, wet_level=0.25)]),
    "double_voice": P("Double Voice (thick)", "Pitch", lambda m: [
        Chorus(rate_hz=0.8, depth=0.35, centre_delay_ms=14, mix=0.5*m+0.2), Compressor(-22, 3)]),
    # Lo-fi / dirt
    "telephone": P("Telephone", "Lo-Fi", lambda m: [
        HighpassFilter(400), LowpassFilter(3400), Distortion(6*m), Gain(3)]),
    "walkie_talkie": P("Walkie-Talkie", "Lo-Fi", lambda m: [
        HighpassFilter(500), LowpassFilter(3000), Bitcrush(bit_depth=10), Distortion(12*m+4),
        Compressor(-25, 8), Gain(2)]),
    "megaphone": P("Megaphone", "Lo-Fi", lambda m: [
        HighpassFilter(600), LowpassFilter(4500), Distortion(15*m+5), PeakFilter(1800, gain_db=6), Gain(1)]),
    "lofi_tape": P("Lo-Fi Tape", "Lo-Fi", lambda m: [
        HighpassFilter(120), LowpassFilter(5500), Bitcrush(bit_depth=14), Chorus(rate_hz=0.3, depth=0.4, mix=0.4),
        Distortion(5*m)]),
    "crunch_8bit": P("8-Bit Crunch", "Lo-Fi", lambda m: [Bitcrush(bit_depth=int(8-3*m)), Limiter(-1)]),
    "vinyl_old": P("Old Vinyl", "Lo-Fi", lambda m: [
        HighpassFilter(200), LowpassFilter(4500), Distortion(8*m), Reverb(room_size=0.2, wet_level=0.1)]),
    # Club / EDM
    "club_pa": P("Club PA System", "Club", lambda m: [
        HighpassFilter(150), PeakFilter(2500, gain_db=4*m), Distortion(3*m),
        Compressor(-20, 6, 2, 60), Reverb(room_size=0.6, wet_level=0.2), Limiter(-0.5)]),
    "phaser_sweep": P("Phaser Sweep", "Club", lambda m: [
        Phaser(rate_hz=0.6, depth=0.9, feedback=0.4, mix=0.6*m+0.2), Reverb(room_size=0.3, wet_level=0.12)]),
    "filter_sweep_low": P("Muffled / Behind the Wall", "Club", lambda m: [
        LowpassFilter(1200-600*m), Reverb(room_size=0.3, wet_level=0.1)]),
    "drop_impact": P("Drop Impact", "Club", lambda m: [
        LowShelfFilter(100, gain_db=9*m), Distortion(8*m), Compressor(-26, 8, 1, 50),
        Delay(delay_seconds=0.18, feedback=0.35, mix=0.25), Reverb(room_size=0.8, wet_level=0.3*m+0.1), Limiter(-0.5)]),
    "sidechain_pump": P("Heavy Limiter Loud", "Club", lambda m: [
        Compressor(-28, 10), Gain(8*m+2), Clipping(-3), Limiter(-0.3)]),
    "whisper_air": P("Airy Whisper", "Club", lambda m: [
        HighpassFilter(250), HighShelfFilter(8000, gain_db=8*m), Reverb(room_size=0.6, wet_level=0.3)]),
    "vocal_widener": P("Wide Vocal", "Club", lambda m: [
        Chorus(rate_hz=0.5, depth=0.4, centre_delay_ms=20, mix=0.5*m+0.1),
        Reverb(room_size=0.4, width=1.0, wet_level=0.15)]),
}

def apply_fx(audio, preset_ids, mix, pitch):
    plugins = []
    if pitch:
        plugins.append(PitchShift(pitch))
    for pid in preset_ids:
        if pid in PRESETS:
            plugins += PRESETS[pid]["fn"](mix)
    plugins.append(Limiter(-1))
    tail = np.zeros(int(SR * 2.0), dtype=np.float32)  # room for reverb/delay tails
    x = np.concatenate([audio, tail])[None, :]
    y = Pedalboard(plugins)(x, SR)
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.89).astype(np.float32)

def encode(y, fmt):
    buf = io.BytesIO()
    with AudioFile(buf, "w", SR, y.shape[0], format=fmt) as f:
        f.write(y)
    buf.seek(0)
    return buf

# ---------- API ----------
@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/api/config")
def config():
    return jsonify({
        "voices": VOICES,
        "presets": [{"id": k, "label": v["label"], "group": v["group"]} for k, v in PRESETS.items()],
        "max_chars": MAX_CHARS,
    })

@app.post("/api/tts")
def tts():
    d = request.get_json(silent=True) or {}
    text = re.sub(r"\s+", " ", str(d.get("text", ""))).strip()
    if not text: return jsonify(error="Text required"), 400
    if len(text) > MAX_CHARS: return jsonify(error=f"Max {MAX_CHARS} characters"), 400
    voice = d.get("voice", "af_heart")
    voice2 = d.get("voice2") or None
    if voice not in VOICES or (voice2 and voice2 not in VOICES):
        return jsonify(error="Unknown voice"), 400
    try:
        blend = min(max(float(d.get("blend", 0)), 0), 1)
        speed = min(max(float(d.get("speed", 1)), 0.5), 2)
        mix = min(max(float(d.get("intensity", 0.6)), 0), 1)
        pitch = min(max(float(d.get("pitch", 0)), -12), 12)
    except ValueError:
        return jsonify(error="Bad number"), 400
    presets = [p for p in (d.get("presets") or [])][:3]
    fmt = "mp3" if d.get("format") == "mp3" else "wav"

    audio = synth(text, voice, voice2, blend, speed)
    y = apply_fx(audio, presets, mix, pitch)
    return send_file(encode(y, fmt), mimetype="audio/mpeg" if fmt == "mp3" else "audio/wav",
                     download_name=f"dj-tag.{fmt}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
