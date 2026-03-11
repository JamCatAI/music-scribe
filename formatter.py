import json
import os
import re
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

console = Console()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:60]


def print_analysis(metadata: dict, analysis: str) -> None:
    console.print()
    console.print(Rule(f"[bold magenta]{metadata['title']}[/bold magenta]"))
    console.print(f"[dim]Artist: {metadata['artist']} · Duration: {metadata['duration']}[/dim]")
    console.print()
    console.print(Markdown(analysis))
    console.print()


def save_analysis(metadata: dict, transcript: str, analysis: str, output_dir: str = "./output") -> str:
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{slugify(metadata['title'])}.md"
    filepath = os.path.join(output_dir, filename)

    date = datetime.now().strftime("%Y-%m-%d")
    content = f"""# {metadata['title']}

**Artist:** {metadata['artist']}
**Duration:** {metadata['duration']}
**URL:** {metadata['url']}
**Analyzed:** {date}

---

{analysis}

---

## Audio Features

{transcript}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def format_substack(metadata: dict, raw_features: dict, analysis: str) -> str:
    """Generate a Substack-ready markdown draft from features + LLM analysis."""
    f = raw_features

    # Header info
    title = metadata.get("title", "Unknown")
    artist = metadata.get("artist", "Unknown")
    duration = metadata.get("duration", "")

    key = f.get("key", "?")
    bpm = f.get("tempo_bpm", "?")
    time_sig = f.get("time_signature", "4/4")
    energy = f.get("energy", "moderate")

    genre_candidates = f.get("genre_candidates", [])
    mood_candidates = f.get("mood_candidates", [])
    genre_str = genre_candidates[0][0] if genre_candidates else "Unknown"
    mood_str = mood_candidates[0][0] if mood_candidates else "Unknown"

    chord_prog = f.get("chord_progression", "")
    roman_prog = f.get("roman_progression", "")

    voiced_ratio = f.get("voiced_ratio", 0)
    vocal_label = "prominent" if voiced_ratio > 0.6 else ("present" if voiced_ratio > 0.3 else "minimal")

    rms = f.get("rms_mean", "?")
    dyn_range = f.get("dynamic_range", "moderate")
    brightness = f.get("brightness", "balanced")
    num_measures = f.get("num_measures", "?")

    # Pull the LLM analysis for the body (strip leading/trailing whitespace)
    analysis_body = analysis.strip()

    date = datetime.now().strftime("%B %d, %Y")

    post = f"""# {title}

*{artist} · {duration} · {date}*

---

**{genre_str} · {mood_str} · {key} · {bpm} BPM · {time_sig}**

---

{analysis_body}

---

## By the Numbers

| Feature | Value |
|---------|-------|
| Key | {key} |
| BPM | {bpm} |
| Time Signature | {time_sig} |
| Energy | {energy} |
| Dynamics | {dyn_range} |
| Brightness | {brightness} |
| Vocals | {vocal_label} ({voiced_ratio:.0%} voiced) |
| Measures analyzed | {num_measures} |

## The Chord Loop

**{chord_prog}**

Roman numerals: *{roman_prog}*

---

*Analyzed with [music-scribe](https://github.com/hoangvu/music-scribe) · Features extracted via librosa · Genre/mood: rule-based estimate*
"""
    return post.strip()


def save_substack(metadata: dict, raw_features: dict, analysis: str, output_dir: str = "./output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{slugify(metadata['title'])}-substack.md"
    filepath = os.path.join(output_dir, filename)
    content = format_substack(metadata, raw_features, analysis)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def save_features_json(metadata: dict, features: dict, output_dir: str = "./output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{slugify(metadata['title'])}.json"
    filepath = os.path.join(output_dir, filename)
    payload = {
        "title": metadata["title"],
        "artist": metadata["artist"],
        "duration": metadata["duration"],
        "url": metadata.get("url"),
        "analyzed": datetime.now().strftime("%Y-%m-%d"),
        "features": features,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath
