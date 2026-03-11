# music-scribe 🎵

**Deep music analysis from any YouTube URL — in your terminal.**

Paste a YouTube link. Get back: key, BPM, chord progression, Roman numeral analysis, time signature, energy, dynamics, vocal presence, timbral fingerprint, and a full LLM-written breakdown — all from raw audio, no lyrics needed.

Works on anything: K-pop, ambient, post-punk, folk, electronic, classical. Any language, any genre.

```
$ python main.py "https://www.youtube.com/watch?v=s1ATTIQrmIQ"

✓ Molchat Doma - Судно (Boris Ryzhy) (3:22)
✓ 120.0 BPM · B minor · high energy · chords: Bm → D → A → E...

━━━━━━━━━━━━━━ Molchat Doma - Судно ━━━━━━━━━━━━━━

## What This Song Is

Cold, mechanical, inevitable. Судно sits at 120 BPM in B minor — a key
historically associated with resignation and dark introspection...

## The Chord Loop

i → III → VII → IV (Bm → D → A → E)
The progression never fully resolves. The major IV (E major) should feel
like release, but at this tempo it feels like a door slamming shut...
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/JamCatAI/music-scribe
cd music-scribe

# 2. Install
pip install -r requirements.txt
brew install ffmpeg   # macOS — or: sudo apt install ffmpeg

# 3. Get a free Gemini API key (no credit card)
# → https://aistudio.google.com/app/apikey

# 4. Set it
echo "GEMINI_API_KEY=your-key-here" > .env

# 5. Analyze
python main.py "https://www.youtube.com/watch?v=..."
```

That's it. Analysis is printed to your terminal and saved as markdown in `./output/`.

---

## What you get

### Audio features (extracted locally via librosa — no API needed for this part)

| Feature | What it tells you |
|---------|------------------|
| BPM + stability | Speed and steadiness of the beat |
| Time signature | 4/4, 3/4 (waltz), 6/8 — detected from beat strength |
| Key + mode | Tonal center and major/minor character |
| Key confidence | How harmonically clear vs. ambiguous |
| Chord progression | Actual chords detected across the song (8s windows) |
| Roman numerals | I–VI–IV–V style analysis relative to the key |
| Harmonic complexity | Simple diatonic vs. chromatic/modal |
| Vocal presence | Voiced ratio — how much of the song has melody |
| Pitch expressiveness | Narrow monotone vs. wide melodic range |
| Energy + loudness | Intensity in dB with dynamic range |
| Timbral mutation | Whether the sound character stays flat or evolves |
| Spectral flux | How restless vs. static the spectrum is |
| Brightness | High-end vs. low-end balance |
| Timeline | All features per 30-second segment |
| Genre + mood | Rule-based estimate (Jazz, Post-Punk, Neo-Soul...) |

### LLM analysis (via Gemini / Claude / OpenAI)

The extracted numbers are sent to an LLM with a musicologist prompt. It writes:
- What the song *is* — character, feel, cultural context
- Why the chord loop works (or doesn't)
- How the structure builds and releases tension
- What makes it distinctive compared to its genre

---

## All commands

```bash
# Single song
python main.py "https://www.youtube.com/watch?v=..."

# Compare two songs side by side
python main.py --compare "URL1" "URL2"

# Local audio file (mp3, wav, flac, m4a)
python main.py "song.mp3"

# Stem separation — vocals / bass / drums / other (uses demucs, slower)
python main.py "URL" --stems

# Save raw features as JSON
python main.py "URL" --json

# Generate a Substack-ready post draft
python main.py "URL" --format substack

# Use Claude or OpenAI instead of Gemini
python main.py "URL" --provider claude
python main.py "URL" --provider openai

# Custom output directory
python main.py "URL" --output-dir ./my-analyses
```

---

## AI providers

| Provider | Cost | Model | Notes |
|----------|------|-------|-------|
| `gemini` (default) | Free — 1,500 req/day | gemini-2.0-flash | Best starting point |
| `claude` | ~$0.01–0.02/song | claude-sonnet-4-6 | Best writing quality |
| `openai` | ~$0.01–0.02/song | gpt-4o | Strong all-rounder |

Get a free Gemini key: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## How it works

```
YouTube URL ──► yt-dlp ──► librosa ──────────────────────────► LLM ──► markdown
                           (BPM, key, chords, timbre,          (Gemini /
                            dynamics, vocal, timeline)          Claude /
                                                                OpenAI)
                           demucs (optional)
                           (vocals / bass / drums / other)
```

All audio processing happens locally. Only the feature text (no audio) is sent to the LLM.

---

## `.env` file

```bash
GEMINI_API_KEY=your-key-here

# Optional — only needed if using --provider claude or --provider openai
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

---

## Requirements

- Python 3.9+
- ffmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg`)
- A free Gemini API key

Optional for stem separation:
- `pip install demucs` (included in requirements.txt)
- A few GB of disk space for the model download on first run

---

## Why music-scribe?

Most music analysis tools either:
- Require you to upload audio to a web service
- Only work on English-language pop
- Give you a genre tag and call it done
- Cost money per request

music-scribe runs locally, works on any language and genre, gives you the actual numbers behind the analysis, and uses the LLM as a musicologist — not a classifier.

---

## License

MIT
