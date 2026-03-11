import os
import glob
import shutil
import subprocess
import tempfile


def separate_stems(audio_path: str) -> tuple[dict, str]:
    """
    Run demucs on audio_path to separate into vocals/bass/drums/other.
    Returns (stems_dict, temp_dir) — caller is responsible for cleanup.
    stems_dict: {"vocals": path, "bass": path, "drums": path, "other": path}
    """
    out_dir = tempfile.mkdtemp(prefix="music-scribe-stems-")

    try:
        result = subprocess.run(
            ["python", "-m", "demucs", "--out", out_dir, audio_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError("demucs not found. Install it: pip install demucs")
    except subprocess.TimeoutExpired:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError("demucs timed out (10 min limit)")

    if result.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(f"demucs failed:\n{result.stderr[-500:]}")

    # demucs outputs to: out_dir/{model_name}/{track_name}/stem.wav
    # Find the stems by globbing — handles any model name
    stem_files = glob.glob(os.path.join(out_dir, "*", "*", "*.wav"))
    if not stem_files:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError("demucs ran but produced no output files")

    stems = {}
    for path in stem_files:
        name = os.path.splitext(os.path.basename(path))[0]  # vocals, bass, drums, other
        stems[name] = path

    return stems, out_dir
