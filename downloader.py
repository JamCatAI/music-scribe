import os
import tempfile
import yt_dlp


def download_audio(source: str) -> tuple[str, dict]:
    """Download from a YouTube URL, or load a local audio file. Returns (audio_path, metadata)."""
    if os.path.exists(source):
        return _load_local(source)
    return _download_youtube(source)


def _load_local(path: str) -> tuple[str, dict]:
    """Return the local file path directly with metadata extracted from the filename."""
    import librosa
    duration_sec = librosa.get_duration(path=path)
    mins, secs = int(duration_sec // 60), int(duration_sec % 60)
    title = os.path.splitext(os.path.basename(path))[0]
    metadata = {
        "title": title,
        "artist": "Unknown",
        "channel": "Local File",
        "duration": f"{mins}:{secs:02d}",
        "upload_date": None,
        "description": None,
        "url": os.path.abspath(path),
    }
    return path, metadata


def _download_youtube(url: str) -> tuple[str, dict]:
    """Download audio from YouTube URL and extract metadata."""
    tmpdir = tempfile.mkdtemp()
    output_path = os.path.join(tmpdir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    audio_file = os.path.join(tmpdir, "audio.mp3")

    metadata = {
        "title": info.get("title", "Unknown"),
        "artist": info.get("artist") or info.get("uploader", "Unknown"),
        "channel": info.get("channel") or info.get("uploader", "Unknown"),
        "duration": info.get("duration_string") or str(info.get("duration", "?")),
        "upload_date": info.get("upload_date", ""),
        "description": (info.get("description") or "")[:500],
        "url": url,
    }

    return audio_file, metadata
