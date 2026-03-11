import whisper
import numpy as np

# Map Whisper language codes to country/region guesses
LANGUAGE_TO_ORIGIN = {
    "en": "English (UK / US / Australia / etc.)",
    "es": "Spanish (Spain / Latin America)",
    "fr": "French (France / Belgium / West Africa)",
    "de": "German (Germany / Austria / Switzerland)",
    "pt": "Portuguese (Brazil / Portugal)",
    "it": "Italian (Italy)",
    "ru": "Russian (Russia / Eastern Europe)",
    "pl": "Polish (Poland)",
    "nl": "Dutch (Netherlands / Belgium)",
    "tr": "Turkish (Turkey)",
    "ar": "Arabic (Middle East / North Africa)",
    "fa": "Persian/Farsi (Iran)",
    "hi": "Hindi (India)",
    "bn": "Bengali (Bangladesh / India)",
    "ur": "Urdu (Pakistan / India)",
    "ta": "Tamil (India / Sri Lanka)",
    "te": "Telugu (India)",
    "ko": "Korean (South Korea)",
    "ja": "Japanese (Japan)",
    "zh": "Chinese (China / Taiwan)",
    "vi": "Vietnamese (Vietnam)",
    "th": "Thai (Thailand)",
    "id": "Indonesian (Indonesia)",
    "ms": "Malay (Malaysia / Indonesia)",
    "sv": "Swedish (Sweden)",
    "no": "Norwegian (Norway)",
    "da": "Danish (Denmark)",
    "fi": "Finnish (Finland)",
    "hu": "Hungarian (Hungary)",
    "cs": "Czech (Czech Republic)",
    "sk": "Slovak (Slovakia)",
    "ro": "Romanian (Romania)",
    "bg": "Bulgarian (Bulgaria)",
    "hr": "Croatian (Croatia)",
    "uk": "Ukrainian (Ukraine)",
    "el": "Greek (Greece)",
    "he": "Hebrew (Israel)",
    "sw": "Swahili (East Africa)",
    "yo": "Yoruba (Nigeria)",
    "am": "Amharic (Ethiopia)",
    "af": "Afrikaans (South Africa)",
    "sq": "Albanian (Albania / Kosovo)",
    "mk": "Macedonian (North Macedonia)",
    "sr": "Serbian (Serbia)",
    "lt": "Lithuanian (Lithuania)",
    "lv": "Latvian (Latvia)",
    "et": "Estonian (Estonia)",
    "ka": "Georgian (Georgia)",
    "hy": "Armenian (Armenia)",
    "az": "Azerbaijani (Azerbaijan)",
    "kk": "Kazakh (Kazakhstan)",
    "uz": "Uzbek (Uzbekistan)",
    "mn": "Mongolian (Mongolia)",
    "ne": "Nepali (Nepal)",
    "si": "Sinhala (Sri Lanka)",
    "km": "Khmer (Cambodia)",
    "lo": "Lao (Laos)",
    "my": "Burmese (Myanmar)",
    "gl": "Galician (Spain/Galicia)",
    "ca": "Catalan (Spain/Catalonia)",
    "eu": "Basque (Spain/France Basque region)",
    "cy": "Welsh (Wales)",
    "ga": "Irish (Ireland)",
    "is": "Icelandic (Iceland)",
    "mt": "Maltese (Malta)",
    "lb": "Luxembourgish (Luxembourg)",
    "bs": "Bosnian (Bosnia)",
    "sl": "Slovenian (Slovenia)",
    "nn": "Nynorsk (Norway)",
}


def detect_language(audio_path: str, model_size: str = "base") -> dict:
    """Detect spoken language from audio using Whisper's language identification."""
    model = whisper.load_model(model_size)

    # Load and pad/trim audio to 30s (Whisper's standard window)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect language probabilities
    _, probs = model.detect_language(mel)

    # Top 3 candidates
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

    top_lang_code, top_prob = top3[0]
    origin = LANGUAGE_TO_ORIGIN.get(top_lang_code, f"Unknown ({top_lang_code})")

    candidates = []
    for code, prob in top3:
        label = LANGUAGE_TO_ORIGIN.get(code, code)
        candidates.append(f"{label} ({prob*100:.1f}%)")

    # Low confidence = probably instrumental/no vocals
    if top_prob < 0.2:
        verdict = "likely instrumental or no clear vocals detected"
    elif top_prob < 0.5:
        verdict = f"possibly {origin} (low confidence — may be minimal or non-standard vocals)"
    else:
        verdict = f"{origin} (confidence: {top_prob*100:.1f}%)"

    return {
        "detected_language": top_lang_code,
        "likely_origin": origin,
        "verdict": verdict,
        "top_candidates": candidates,
    }


def transcribe(audio_path: str, model_size: str = "base") -> str:
    """Transcribe audio file to text using Whisper."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"].strip()
