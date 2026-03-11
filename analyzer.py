import os
import anthropic
import openai
from google import genai


PROMPT_TEMPLATE = """You are an expert music critic and cultural analyst. Analyze this song deeply using the raw audio features extracted by librosa.

## Song Info
Title: {title}
Artist: {artist}
Duration: {duration}
Description: {description}

{audio_features}

---

Provide a thorough analysis with these four sections:

### 1. Language & Origin
Based on the artist name, song title, and your own knowledge — what language are the vocals likely in? What country or region is this artist from? Make your best guess and explain your reasoning. Be direct even if speculative.

### 2. Sonic Breakdown
Interpret the raw audio features above — what do they tell us about the sound? Discuss the tempo feel, key/mood relationship, energy arc, textural character, and timbral fingerprint. Be specific and musical.

### 3. Cultural Context
Who is this artist? What era/movement do they belong to? What genre or scene does this fit into? What influences or communities shaped this sound?

### 4. Emotional & Artistic Character
What emotion or atmosphere does this music create? What might this song be used for (film score, workout, meditation, club, etc.)? What makes it interesting or distinctive as a piece of music?
"""

COMPARE_TEMPLATE = """You are an expert music critic and cultural analyst. Compare these two songs using their raw audio features.

## Song A: {title_a}
Artist: {artist_a} · Duration: {duration_a}

{features_a}

---

## Song B: {title_b}
Artist: {artist_b} · Duration: {duration_b}

{features_b}

---

Provide a deep comparative analysis with these four sections:

### 1. Origins & Languages
For each song — what language are the vocals likely in, and where is the artist from? Use your own knowledge, be direct even if speculative.

### 2. Sonic Comparison
Compare the two songs across: tempo feel, key and mood, energy arc, use of silence, timbral character, rhythmic regularity, and how the sound evolves over the timeline. What does each do that the other doesn't?

### 3. Cultural Distance & Connection
How far apart are these two songs culturally, historically, geographically? What scene or tradition does each belong to? Despite their differences, what structural or symbolic patterns do they share — if any?

### 4. The Surprising Link
What is the single most unexpected thing these two songs have in common when you look at the raw data? This is the insight a music journalist would build an article around.
"""


def analyze(audio_features: str, metadata: dict, provider: str = "gemini") -> str:
    """Send audio features + metadata to LLM for deep music analysis."""
    prompt = PROMPT_TEMPLATE.format(
        title=metadata["title"],
        artist=metadata["artist"],
        duration=metadata["duration"],
        description=metadata["description"] or "N/A",
        audio_features=audio_features,
    )
    return _call_provider(prompt, provider)


def compare(features_a: str, metadata_a: dict, features_b: str, metadata_b: dict, provider: str = "gemini") -> str:
    """Send two songs' features to LLM for comparative analysis."""
    prompt = COMPARE_TEMPLATE.format(
        title_a=metadata_a["title"], artist_a=metadata_a["artist"], duration_a=metadata_a["duration"],
        features_a=features_a,
        title_b=metadata_b["title"], artist_b=metadata_b["artist"], duration_b=metadata_b["duration"],
        features_b=features_b,
    )
    return _call_provider(prompt, provider)


def _call_provider(prompt: str, provider: str) -> str:
    if provider == "claude":
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    elif provider == "openai":
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'openai', or 'gemini'.")
