import numpy as np
import librosa

# --- Chord templates (24: 12 major + 12 minor) ---
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _build_chord_templates() -> dict:
    templates = {}
    for i, note in enumerate(_NOTE_NAMES):
        major = np.zeros(12)
        major[i] = major[(i + 4) % 12] = major[(i + 7) % 12] = 1.0
        templates[note] = major / np.linalg.norm(major)
        minor = np.zeros(12)
        minor[i] = minor[(i + 3) % 12] = minor[(i + 7) % 12] = 1.0
        templates[f"{note}m"] = minor / np.linalg.norm(minor)
    return templates


_CHORD_TEMPLATES = _build_chord_templates()


def _detect_chord(chroma_mean: np.ndarray) -> str:
    norm = np.linalg.norm(chroma_mean)
    if norm < 1e-6:
        return "—"
    chroma_norm = chroma_mean / norm
    return max(_CHORD_TEMPLATES, key=lambda c: float(np.dot(chroma_norm, _CHORD_TEMPLATES[c])))


# --- Roman numeral lookup tables ---
_MAJOR_NUMERALS = {0: 'I', 2: 'ii', 4: 'iii', 5: 'IV', 7: 'V', 9: 'vi', 11: 'vii°'}
_MINOR_NUMERALS = {0: 'i', 2: 'ii°', 3: 'III', 5: 'iv', 7: 'v', 8: 'VI', 10: 'VII'}


def _chord_to_roman(chord_name: str, key_root_idx: int, mode: str) -> str:
    """Convert chord name (e.g. 'F#m') to Roman numeral relative to the key."""
    if chord_name in ('—', ''):
        return '—'
    is_minor_chord = chord_name.endswith('m')
    chord_root_name = chord_name[:-1] if is_minor_chord else chord_name
    if chord_root_name not in _NOTE_NAMES:
        return chord_name
    chord_root_idx = _NOTE_NAMES.index(chord_root_name)
    interval = (chord_root_idx - key_root_idx) % 12

    numeral = (_MINOR_NUMERALS if mode == 'minor' else _MAJOR_NUMERALS).get(interval)
    if numeral is None:
        # Chromatic/borrowed — show with ♭ prefix
        flat_intervals = {1, 3, 6, 8, 10}
        prefix = '♭' if interval in flat_intervals else '♯'
        base = _NOTE_NAMES[interval]
        numeral = f"{prefix}{'VII' if interval == 10 else 'VI' if interval == 8 else 'V' if interval == 6 else base}"

    # Adjust quality: if actual chord quality differs from expected scale degree quality
    expected_minor = numeral[0].islower() or '°' in numeral
    if is_minor_chord and not expected_minor:
        numeral = numeral.lower()        # major degree played as minor (borrowed)
    elif not is_minor_chord and expected_minor and numeral != 'vii°':
        numeral = numeral.upper()        # minor degree played as major (e.g. v→V dominant)

    return numeral


def _estimate_time_signature(beats: np.ndarray, y: np.ndarray, sr: int) -> tuple[str, int]:
    """Estimate 3/4 vs 4/4 from beat strength periodicity."""
    if len(beats) < 6:
        return "4/4", 4
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_env = onset_env[np.minimum(beats, len(onset_env) - 1)]

    def accent_score(be, period):
        if len(be) < period * 2:
            return 0.0
        beat1 = be[::period]
        others = np.concatenate([be[i::period] for i in range(1, period)])
        return float(np.mean(beat1) / (np.mean(others) + 1e-8))

    score_4 = accent_score(beat_env, 4)
    score_3 = accent_score(beat_env, 3)
    if score_3 > score_4 * 1.15:
        return "3/4 (triple meter / waltz)", 3
    return "4/4 (common time)", 4


def classify_genre_mood(f: dict) -> dict:
    """Rule-based genre and mood estimation from extracted features."""
    bpm = f['tempo_bpm']
    energy = f['rms_mean']
    voiced = f['voiced_ratio']
    centroid = f['spectral_centroid_hz']
    density = f['onset_rate_per_sec']
    harmony = f['harmonic_complexity']
    rhythm = f['rhythmic_regularity']
    mode = 'minor' if 'minor' in f['key'] else 'major'
    silence = f['silence_ratio']
    timbral = f['timbral_variance']
    flux = f['flux_norm']
    character = f['character']
    dynamic = f['rms_std']
    pitch_range = f.get('pitch_range_cents', 0)

    def score(rules):
        return sum(v for _, v in rules if _)

    genres = {
        'Electronic/Dance': score([
            (118 <= bpm <= 148, 2), (rhythm == 'on-grid (very quantized)', 2),
            (density > 3, 1), (voiced < 0.3, 1), (centroid > 2000, 1),
        ]),
        'Pop': score([
            (90 <= bpm <= 135, 1), (voiced > 0.4, 2), (energy > 0.03, 1),
            (harmony in ('simple', 'moderate'), 1), (density > 2, 1),
        ]),
        'Rock': score([
            (100 <= bpm <= 165, 1), (character == 'rhythm/percussion-driven', 2),
            (energy > 0.05, 2), (centroid < 3000, 1),
        ]),
        'Classical/Orchestral': score([
            (dynamic > 0.04, 2), (character == 'heavily melodic/harmonic', 2),
            (harmony == 'complex/chromatic', 1), (centroid > 2000, 1),
        ]),
        'Folk/Acoustic': score([
            (bpm < 110, 1), (density < 3, 1), (voiced > 0.4, 2),
            (centroid < 2500, 1), (harmony in ('simple', 'moderate'), 1),
        ]),
        'Jazz': score([
            (harmony == 'complex/chromatic', 2),
            (rhythm not in ('on-grid (very quantized)',), 1),
            (70 <= bpm <= 140, 1), (character != 'rhythm/percussion-driven', 1),
        ]),
        'Ambient/Drone': score([
            (density < 1.5, 2), (silence > 0.15, 2),
            (energy < 0.03, 2), (timbral < 10, 1),
        ]),
        'Hip-Hop/R&B': score([
            (75 <= bpm <= 105, 2), (character == 'rhythm/percussion-driven', 2),
            (centroid < 2500, 1), (rhythm == 'off-grid/syncopated (floats around the beat)', 1),
        ]),
    }
    moods = {
        'Energetic': score([
            (energy > 0.05, 2), (bpm > 120, 1), (density > 3, 1), (silence < 0.1, 1),
        ]),
        'Melancholic': score([
            (mode == 'minor', 2), (dynamic > 0.03, 1),
            (pitch_range > 400, 1), (bpm < 120, 1),
        ]),
        'Peaceful/Calm': score([
            (energy < 0.03, 2), (density < 2, 2), (silence > 0.15, 1), (bpm < 90, 1),
        ]),
        'Tense/Anxious': score([
            (harmony == 'complex/chromatic', 2), (flux > 1.0, 1),
            (mode == 'minor', 1), (dynamic > 0.04, 1),
        ]),
        'Happy/Upbeat': score([
            (mode == 'major', 2), (bpm > 100, 1),
            (energy > 0.03, 1), (harmony == 'simple', 1),
        ]),
        'Aggressive/Intense': score([
            (energy > 0.07, 2), (density > 4, 1),
            (character == 'rhythm/percussion-driven', 1), (harmony == 'complex/chromatic', 1),
        ]),
        'Romantic/Intimate': score([
            (voiced > 0.5, 2), (bpm < 100, 1),
            (dynamic > 0.02, 1), (pitch_range > 800, 1),
        ]),
    }

    top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
    top_moods = sorted(moods.items(), key=lambda x: x[1], reverse=True)
    return {
        'genre_candidates': [(g, s) for g, s in top_genres if s > 0][:3],
        'mood_candidates': [(m, s) for m, s in top_moods if s > 0][:3],
    }


def extract_features(audio_path: str, segment_sec: int = 30) -> dict:
    """Extract musical features from audio using librosa, globally and per time segment."""
    y, sr = librosa.load(audio_path)  # full song
    duration = len(y) / sr

    # --- Global features ---

    # Harmonic / percussive split
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(librosa.feature.rms(y=y_harmonic)[0]))
    percussive_energy = float(np.mean(librosa.feature.rms(y=y_percussive)[0]))
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)
    character = "heavily melodic/harmonic" if harmonic_ratio > 0.7 else "balanced melodic+rhythmic" if harmonic_ratio > 0.4 else "rhythm/percussion-driven"

    # Tempo + stability
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_ioi_std = None
    beat_times = np.array([])
    if len(beats) > 2:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        ioi = np.diff(beat_times)
        beat_ioi_std = round(float(np.std(ioi)), 3)
        tempo_stability = "very steady" if beat_ioi_std < 0.02 else "mostly steady" if beat_ioi_std < 0.06 else "loose/rubato"
    else:
        tempo_stability = "undetectable (free tempo or ambient)"

    # Time signature + beat grid
    time_signature, beats_per_measure = _estimate_time_signature(beats, y, sr)
    num_measures = len(beat_times) // beats_per_measure if len(beat_times) > 0 else 0
    avg_measure_dur = round(float(np.mean(np.diff(beat_times[::beats_per_measure]))), 2) if len(beat_times) >= beats_per_measure * 2 else None

    # Rhythmic regularity — do onsets align with the beat grid?
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    avg_beat_deviation = None
    if len(beats) > 1 and len(onsets) > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        onset_beat_dists = [min(abs(o - beat_times)) for o in onsets]
        avg_beat_deviation = round(float(np.mean(onset_beat_dists)), 3)
        rhythmic_regularity = "on-grid (very quantized)" if avg_beat_deviation < 0.05 else "mostly on-grid" if avg_beat_deviation < 0.12 else "off-grid/syncopated (floats around the beat)"
    else:
        rhythmic_regularity = "indeterminate"

    # Key
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    root_idx = int(np.argmax(chroma_mean))
    dominant_note = _NOTE_NAMES[root_idx]
    major_third = chroma_mean[(root_idx + 4) % 12]
    minor_third = chroma_mean[(root_idx + 3) % 12]
    mode = "major" if major_third > minor_third else "minor"
    key_confidence = round(float(chroma_mean[root_idx] / (chroma_mean.sum() + 1e-8)), 3)
    key_confidence_label = "strong/clear" if key_confidence > 0.12 else "moderate" if key_confidence > 0.09 else "ambiguous/chromatic"
    chroma_norm_global = chroma_mean / (chroma_mean.sum() + 1e-8)
    chroma_entropy = round(float(-np.sum(chroma_norm_global * np.log2(chroma_norm_global + 1e-8))), 2)
    harmonic_complexity = "simple" if chroma_entropy < 2.8 else "moderate" if chroma_entropy < 3.2 else "complex/chromatic"

    # Chord progression — sample every 8 seconds, deduplicate
    chord_window = int(8 * sr)
    chord_seq = []
    for i in range(0, len(y_harmonic), chord_window):
        seg = y_harmonic[i:i + chord_window]
        if len(seg) < sr:
            break
        c = librosa.feature.chroma_cqt(y=seg, sr=sr).mean(axis=1)
        chord_seq.append(_detect_chord(c))
    chord_progression_list = []
    for ch in chord_seq:
        if not chord_progression_list or ch != chord_progression_list[-1]:
            chord_progression_list.append(ch)
    chord_progression = " → ".join(chord_progression_list[:16])
    if len(chord_progression_list) > 16:
        chord_progression += " ..."

    # Roman numeral progression
    roman_list = [_chord_to_roman(c, root_idx, mode) for c in chord_progression_list[:16]]
    roman_progression = " → ".join(roman_list)
    if len(chord_progression_list) > 16:
        roman_progression += " ..."

    # Pitch contour
    f0, voiced_flag, _ = librosa.pyin(
        y_harmonic,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_ratio = round(len(voiced_f0) / max(len(f0), 1), 2) if len(f0) > 0 else 0.0
    if len(voiced_f0) > 10:
        f0_cents = 1200 * np.log2(voiced_f0 / (np.mean(voiced_f0) + 1e-8) + 1e-8)
        pitch_range_cents = round(float(np.percentile(f0_cents, 95) - np.percentile(f0_cents, 5)), 0)
        pitch_expressiveness = "very expressive (large pitch range)" if pitch_range_cents > 1200 else "moderate pitch movement" if pitch_range_cents > 400 else "narrow/monotone pitch"
        vocal_presence = "strong vocals detected" if voiced_ratio > 0.4 else "some vocals/melody" if voiced_ratio > 0.15 else "mostly instrumental or whispered"
    else:
        pitch_range_cents = 0.0
        pitch_expressiveness = "no clear pitched melody detected"
        vocal_presence = "likely instrumental"

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = round(float(np.mean(rms)), 4)
    energy_std = round(float(np.std(rms)), 4)
    energy_label = "high" if energy_mean > 0.05 else "medium" if energy_mean > 0.02 else "low"
    dynamic_range = "wide dynamics" if energy_std > 0.03 else "moderate dynamics" if energy_std > 0.01 else "compressed/flat"
    loudness_db = round(float(librosa.amplitude_to_db(np.array([energy_mean]))[0]), 1)

    # Silence ratio
    silence_threshold = energy_mean * 0.1
    silence_frames = np.sum(rms < silence_threshold)
    silence_ratio = round(float(silence_frames / len(rms)), 2)
    silence_label = "lots of silence/space" if silence_ratio > 0.3 else "some breathing room" if silence_ratio > 0.1 else "densely filled (little silence)"

    # Spectral flux
    spec = np.abs(librosa.stft(y))
    flux = np.mean(np.diff(spec, axis=1) ** 2)
    flux_norm = round(float(flux / (np.mean(spec) ** 2 + 1e-8)), 3)
    spectral_flux = "highly restless/evolving (spectrum changes rapidly)" if flux_norm > 0.5 else "moderately evolving" if flux_norm > 0.15 else "static/stable (spectrum barely changes)"

    # Timbral variance
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = [round(x, 2) for x in mfcc.mean(axis=1).tolist()]
    mfcc_std_per_coef = mfcc.std(axis=1)
    timbral_variance = round(float(np.mean(mfcc_std_per_coef[1:])), 2)
    timbral_mutation = "highly mutating timbre (sound character shifts significantly)" if timbral_variance > 20 else "moderate timbral variation" if timbral_variance > 10 else "stable timbre (consistent sound throughout)"

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    brightness = "bright/airy" if centroid_mean > 3000 else "warm/mid-focused" if centroid_mean > 1500 else "dark/bass-heavy"
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = float(np.mean(contrast))
    punch = "punchy/defined" if contrast_mean > 20 else "dense/blended"
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rolloff_mean = round(float(np.mean(rolloff)))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = round(float(np.mean(zcr)), 4)
    texture = "noisy/percussive" if zcr_mean > 0.1 else "smooth/tonal"

    # Onset density
    onset_rate = round(len(onsets) / (len(y) / sr), 2)
    density = "very dense" if onset_rate > 5 else "dense" if onset_rate > 3 else "moderate" if onset_rate > 1 else "sparse/minimal"

    # --- Per-segment features ---
    segments = []
    hop = int(segment_sec * sr)
    num_segments = int(np.ceil(len(y) / hop))

    for i in range(num_segments):
        seg = y[i * hop: (i + 1) * hop]
        if len(seg) < sr:
            break

        t_start = i * segment_sec
        t_end = min(t_start + segment_sec, int(duration))
        label = f"{t_start // 60}:{t_start % 60:02d}–{t_end // 60}:{t_end % 60:02d}"

        seg_rms = float(np.mean(librosa.feature.rms(y=seg)[0]))
        seg_db = round(float(librosa.amplitude_to_db(np.array([seg_rms]))[0]), 1)
        seg_energy = "high" if seg_rms > 0.05 else "medium" if seg_rms > 0.02 else "low"

        seg_centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)[0]))
        seg_brightness = "bright" if seg_centroid > 3000 else "warm/mid" if seg_centroid > 1500 else "dark"

        seg_onsets = librosa.onset.onset_detect(y=seg, sr=sr)
        seg_onset_rate = len(seg_onsets) / (len(seg) / sr)
        seg_density = "very dense" if seg_onset_rate > 5 else "dense" if seg_onset_rate > 3 else "moderate" if seg_onset_rate > 1 else "sparse"

        seg_harm, seg_perc = librosa.effects.hpss(seg)
        seg_h_energy = float(np.mean(librosa.feature.rms(y=seg_harm)[0]))
        seg_p_energy = float(np.mean(librosa.feature.rms(y=seg_perc)[0]))
        seg_ratio = seg_h_energy / (seg_h_energy + seg_p_energy + 1e-8)
        seg_char = "melodic" if seg_ratio > 0.6 else "balanced" if seg_ratio > 0.4 else "rhythmic"

        seg_rms_frames = librosa.feature.rms(y=seg)[0]
        seg_silence = float(np.sum(seg_rms_frames < seg_rms * 0.1) / len(seg_rms_frames))
        seg_space = "spacious" if seg_silence > 0.3 else "some space" if seg_silence > 0.1 else "filled"

        seg_spec = np.abs(librosa.stft(seg))
        seg_flux = float(np.mean(np.diff(seg_spec, axis=1) ** 2) / (np.mean(seg_spec) ** 2 + 1e-8))
        seg_motion = "restless" if seg_flux > 0.5 else "evolving" if seg_flux > 0.15 else "static"

        # Chord for this segment
        seg_chroma = librosa.feature.chroma_cqt(y=seg_harm, sr=sr).mean(axis=1)
        seg_chord = _detect_chord(seg_chroma)

        segments.append({
            "timestamp": label,
            "energy": seg_energy,
            "loudness_db": seg_db,
            "brightness": seg_brightness,
            "density": seg_density,
            "character": seg_char,
            "space": seg_space,
            "spectral_motion": seg_motion,
            "chord": seg_chord,
        })

    # Genre + mood classification
    genre_mood = classify_genre_mood({
        'tempo_bpm': round(float(tempo), 1), 'rms_mean': energy_mean, 'voiced_ratio': voiced_ratio,
        'spectral_centroid_hz': round(centroid_mean), 'onset_rate_per_sec': onset_rate,
        'harmonic_complexity': harmonic_complexity, 'rhythmic_regularity': rhythmic_regularity,
        'key': f"{dominant_note} {mode}", 'silence_ratio': silence_ratio,
        'timbral_variance': timbral_variance, 'flux_norm': flux_norm, 'character': character,
        'rms_std': energy_std, 'pitch_range_cents': pitch_range_cents,
    })

    return {
        "duration_sec": round(duration, 1),
        "tempo_bpm": round(float(tempo), 1),
        "tempo_stability": tempo_stability,
        "beat_ioi_std": beat_ioi_std,
        "time_signature": time_signature,
        "num_measures": num_measures,
        "avg_measure_dur_sec": avg_measure_dur,
        "rhythmic_regularity": rhythmic_regularity,
        "avg_beat_deviation": avg_beat_deviation,
        "key": f"{dominant_note} {mode}",
        "key_confidence": key_confidence_label,
        "key_confidence_raw": key_confidence,
        "harmonic_complexity": harmonic_complexity,
        "chroma_entropy": chroma_entropy,
        "chord_progression": chord_progression,
        "roman_progression": roman_progression,
        "character": character,
        "harmonic_ratio": round(harmonic_ratio, 2),
        "vocal_presence": vocal_presence,
        "voiced_ratio": voiced_ratio,
        "pitch_expressiveness": pitch_expressiveness,
        "pitch_range_cents": pitch_range_cents,
        "energy": energy_label,
        "rms_mean": energy_mean,
        "rms_std": energy_std,
        "loudness_db": loudness_db,
        "dynamic_range": dynamic_range,
        "silence": silence_label,
        "silence_ratio": silence_ratio,
        "spectral_flux": spectral_flux,
        "flux_norm": flux_norm,
        "timbral_mutation": timbral_mutation,
        "timbral_variance": timbral_variance,
        "brightness": brightness,
        "spectral_punch": punch,
        "texture": texture,
        "zcr_mean": zcr_mean,
        "arrangement_density": density,
        "spectral_centroid_hz": round(centroid_mean),
        "spectral_rolloff_hz": rolloff_mean,
        "onset_rate_per_sec": onset_rate,
        "mfcc_timbre": mfcc_means,
        "timeline": segments,
        "genre_candidates": genre_mood["genre_candidates"],
        "mood_candidates": genre_mood["mood_candidates"],
    }


def extract_stem_features(stems: dict) -> dict:
    """Extract focused features from each demucs stem."""
    results = {}

    for stem_name, stem_path in stems.items():
        y, sr = librosa.load(stem_path)

        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        loudness_db = round(float(librosa.amplitude_to_db(np.array([energy_mean]))[0]), 1)
        energy_label = "high" if energy_mean > 0.05 else "medium" if energy_mean > 0.02 else "low"
        dynamic_range = "wide" if energy_std > 0.03 else "moderate" if energy_std > 0.01 else "compressed"

        centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        brightness = "bright" if centroid_mean > 3000 else "warm/mid" if centroid_mean > 1500 else "dark/bass-heavy"

        data = {
            "energy": energy_label,
            "rms_mean": round(energy_mean, 4),
            "loudness_db": loudness_db,
            "dynamic_range": dynamic_range,
            "brightness": brightness,
            "centroid_hz": round(centroid_mean),
        }

        if stem_name == "vocals":
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            voiced_ratio = len(voiced_f0) / max(len(f0), 1)
            if len(voiced_f0) > 10:
                f0_cents = 1200 * np.log2(voiced_f0 / (np.mean(voiced_f0) + 1e-8) + 1e-8)
                pitch_range = float(np.percentile(f0_cents, 95) - np.percentile(f0_cents, 5))
                pitch_expr = "very expressive" if pitch_range > 1200 else "moderate" if pitch_range > 400 else "narrow/monotone"
            else:
                pitch_range = 0.0
                pitch_expr = "minimal pitched content"
            data["voiced_ratio"] = round(voiced_ratio, 2)
            data["pitch_range_cents"] = round(pitch_range, 0)
            data["pitch_expressiveness"] = pitch_expr

        elif stem_name == "drums":
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
            onset_rate = round(len(onsets) / (len(y) / sr), 2)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            data["onset_rate"] = onset_rate
            data["bpm"] = round(float(tempo), 1)

        elif stem_name == "bass":
            rolloff_50 = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.5)[0]))
            data["dominant_freq_hz"] = round(rolloff_50)

        results[stem_name] = data

    return results


def stems_to_text(stem_features: dict) -> str:
    """Format stem features as a readable section for the LLM."""
    lines = ["## Stem Analysis (demucs source separation)\n"]
    for stem in ["vocals", "drums", "bass", "other"]:
        if stem not in stem_features:
            continue
        d = stem_features[stem]
        parts = [
            f"energy: {d['energy']} (rms: {d['rms_mean']}, {d['loudness_db']} dB)",
            f"brightness: {d['brightness']} ({d['centroid_hz']} Hz)",
            f"dynamics: {d['dynamic_range']}",
        ]
        if stem == "vocals":
            parts.append(
                f"pitch: {d.get('pitch_expressiveness', 'n/a')} "
                f"(voiced: {d.get('voiced_ratio', 0):.0%}, range: {d.get('pitch_range_cents', 0):.0f}¢)"
            )
        elif stem == "drums":
            parts.append(
                f"onset rate: {d.get('onset_rate', 'n/a')}/sec, "
                f"BPM: {d.get('bpm', 'n/a')}"
            )
        elif stem == "bass":
            parts.append(f"dominant freq: {d.get('dominant_freq_hz', 'n/a')} Hz")
        lines.append(f"- **{stem.capitalize()}:** {', '.join(parts)}")
    return "\n".join(lines)


def features_to_text(features: dict) -> str:
    """Convert extracted features into a readable description for LLM."""
    timeline_lines = []
    for seg in features["timeline"]:
        timeline_lines.append(
            f"  - **{seg['timestamp']}** — {seg['loudness_db']} dB, chord: {seg['chord']}, "
            f"energy: {seg['energy']}, tone: {seg['brightness']}, density: {seg['density']}, "
            f"character: {seg['character']}, space: {seg['space']}, motion: {seg['spectral_motion']}"
        )
    timeline_text = "\n".join(timeline_lines)

    beat_dev = f" (avg beat deviation: {features['avg_beat_deviation']}s)" if features.get("avg_beat_deviation") else ""
    beat_std = f", IOI σ={features['beat_ioi_std']}s" if features.get("beat_ioi_std") else ""
    measure_info = f", {features['num_measures']} measures" if features.get("num_measures") else ""
    measure_dur = f" (~{features['avg_measure_dur_sec']}s/measure)" if features.get("avg_measure_dur_sec") else ""

    genre_str = ", ".join(f"{g} ({s})" for g, s in features.get("genre_candidates", []))
    mood_str = ", ".join(f"{m} ({s})" for m, s in features.get("mood_candidates", []))

    return f"""## Audio Feature Analysis (extracted from raw audio via librosa)

### Genre & Mood (rule-based estimate)
- **Genre:** {genre_str or 'undetermined'}
- **Mood:** {mood_str or 'undetermined'}

### Rhythm & Tempo
- **BPM:** {features['tempo_bpm']} — {features['tempo_stability']}{beat_std}
- **Time Signature:** {features.get('time_signature', '4/4')}{measure_info}{measure_dur}
- **Rhythmic Regularity:** {features['rhythmic_regularity']}{beat_dev}
- **Arrangement Density:** {features['arrangement_density']} ({features['onset_rate_per_sec']} events/sec)

### Harmony & Tonality
- **Key:** {features['key']} (confidence: {features['key_confidence_raw']} — {features['key_confidence']}, entropy: {features['chroma_entropy']} — {features['harmonic_complexity']})
- **Chord Progression:** {features['chord_progression']}
- **Roman Numerals:** {features.get('roman_progression', '—')}

### Voice & Melody
- **Vocal Presence:** {features['vocal_presence']} (voiced ratio: {features['voiced_ratio']:.0%})
- **Pitch Expressiveness:** {features['pitch_expressiveness']} (pitch range: {features['pitch_range_cents']:.0f}¢)

### Energy & Dynamics
- **Energy:** {features['energy']} (rms: {features['rms_mean']}, {features['loudness_db']} dB, dynamic σ={features['rms_std']} — {features['dynamic_range']})
- **Silence Ratio:** {features['silence_ratio']} — {features['silence']}

### Sonic Character
- **Tonal Character:** {features['character']} (harmonic ratio: {features['harmonic_ratio']})
- **Timbral Mutation:** {features['timbral_mutation']} (variance: {features['timbral_variance']})
- **Spectral Flux:** {features['flux_norm']} — {features['spectral_flux']}
- **Brightness:** {features['brightness']} (centroid: {features['spectral_centroid_hz']} Hz, rolloff: {features['spectral_rolloff_hz']} Hz)
- **Texture:** {features['texture']} (zcr: {features['zcr_mean']})
- **Spectral Punch:** {features['spectral_punch']}

### Timeline (per 30-second segment)
{timeline_text}

### Timbre Fingerprint (MFCC)
{features['mfcc_timbre']}
"""
