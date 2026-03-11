import argparse
import os
import sys
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()

PROVIDERS = ["claude", "openai", "gemini"]
API_KEY_MAP = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def validate_key(provider: str):
    key = API_KEY_MAP[provider]
    if not os.environ.get(key):
        console.print(f"[red]Error: {key} not set. Add it to your .env file.[/red]")
        console.print(f"[dim]Get a free Gemini key at: https://aistudio.google.com/app/apikey[/dim]" if provider == "gemini" else "")
        sys.exit(1)


def process_song(url: str, provider: str, use_stems: bool = False) -> tuple[dict, str, dict]:
    """Download + extract features for a single URL. Returns (metadata, features_text, raw_features)."""
    import shutil
    from downloader import download_audio
    from extractor import extract_features, features_to_text, extract_stem_features, stems_to_text

    is_local = os.path.exists(url)
    console.print(f"\n[bold cyan]{'Loading' if is_local else 'Downloading'}:[/bold cyan] {url}")
    try:
        audio_path, metadata = download_audio(url)
    except Exception as e:
        console.print(f"[red]{'Load' if is_local else 'Download'} failed: {e}[/red]")
        if not is_local:
            console.print("[dim]Check the URL is public and available in your region.[/dim]")
        sys.exit(1)
    console.print(f"[green]✓[/green] {metadata['title']} ({metadata['duration']})")

    console.print(f"[bold cyan]Extracting audio features...[/bold cyan]")
    try:
        raw_features = extract_features(audio_path)
    except Exception as e:
        console.print(f"[red]Feature extraction failed: {e}[/red]")
        sys.exit(1)
    features_text = features_to_text(raw_features)
    console.print(
        f"[green]✓[/green] {raw_features['tempo_bpm']} BPM · {raw_features['key']} · "
        f"{raw_features['energy']} energy · {raw_features['duration_sec']}s analyzed · "
        f"chords: {raw_features['chord_progression'][:40]}{'...' if len(raw_features['chord_progression']) > 40 else ''}"
    )

    if use_stems:
        from separator import separate_stems
        console.print(f"[bold cyan]Separating stems (demucs — this takes a few minutes)...[/bold cyan]")
        stems_dir = None
        try:
            stems, stems_dir = separate_stems(audio_path)
            console.print(f"[green]✓[/green] Stems separated: {', '.join(stems.keys())}")
            stem_features = extract_stem_features(stems)
            features_text += "\n\n" + stems_to_text(stem_features)
            raw_features["stems"] = stem_features
            console.print(f"[green]✓[/green] Stem features extracted")
        except Exception as e:
            console.print(f"[yellow]⚠ Stem separation failed: {e}[/yellow]")
            console.print("[dim]Continuing without stem analysis.[/dim]")
        finally:
            if stems_dir:
                shutil.rmtree(stems_dir, ignore_errors=True)

    if not is_local:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return metadata, features_text, raw_features


def main():
    parser = argparse.ArgumentParser(
        description="Analyze music from YouTube using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://youtube.com/watch?v=..."
  python main.py "https://youtube.com/watch?v=..." --provider gemini
  python main.py --compare "https://youtube.com/watch?v=..." "https://youtube.com/watch?v=..."
        """
    )
    parser.add_argument("urls", nargs="+", help="YouTube URL(s). Pass two with --compare.")
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="gemini",
        help="AI provider (default: gemini — free tier, no credit card)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two songs side by side (requires exactly 2 URLs)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory to save analysis markdown files (default: ./output)",
    )
    parser.add_argument(
        "--stems",
        action="store_true",
        help="Separate song into vocals/bass/drums/other with demucs before analysis (slower, requires pip install demucs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also save raw audio features as a .json file in the output directory",
    )
    parser.add_argument(
        "--format",
        choices=["default", "substack"],
        default="default",
        help="Output format: 'substack' generates a copy-paste ready Substack post draft (default: default)",
    )
    args = parser.parse_args()

    # Validate
    validate_key(args.provider)

    if args.compare and len(args.urls) != 2:
        console.print("[red]--compare requires exactly 2 URLs.[/red]")
        sys.exit(1)

    if not args.compare and len(args.urls) != 1:
        console.print("[red]Pass exactly 1 URL, or use --compare with 2 URLs.[/red]")
        sys.exit(1)

    from analyzer import analyze, compare
    from formatter import print_analysis, save_analysis, save_features_json, save_substack

    if args.compare:
        # --- Compare mode ---
        metadata_a, features_a, raw_a = process_song(args.urls[0], args.provider, use_stems=args.stems)
        metadata_b, features_b, raw_b = process_song(args.urls[1], args.provider, use_stems=args.stems)

        console.print(f"\n[bold cyan]Comparing with {args.provider}...[/bold cyan]")
        try:
            analysis = compare(features_a, metadata_a, features_b, metadata_b, provider=args.provider)
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            sys.exit(1)
        console.print(f"[green]✓[/green] Comparison complete")

        combined_metadata = {
            "title": f"{metadata_a['title']} vs {metadata_b['title']}",
            "artist": f"{metadata_a['artist']} · {metadata_b['artist']}",
            "duration": f"{metadata_a['duration']} / {metadata_b['duration']}",
            "url": f"{args.urls[0]} | {args.urls[1]}",
        }
        combined_features = f"## Song A: {metadata_a['title']}\n{features_a}\n\n## Song B: {metadata_b['title']}\n{features_b}"
        print_analysis(combined_metadata, analysis)
        filepath = save_analysis(combined_metadata, combined_features, analysis, output_dir=args.output_dir)
        if args.json:
            jpath = save_features_json(combined_metadata, {"song_a": raw_a, "song_b": raw_b}, output_dir=args.output_dir)
            console.print(f"[dim]JSON saved to: {jpath}[/dim]")

    else:
        # --- Single song mode ---
        metadata, features_text, raw_features = process_song(args.urls[0], args.provider, use_stems=args.stems)

        console.print(f"[bold cyan]Analyzing with {args.provider}...[/bold cyan]")
        try:
            analysis = analyze(features_text, metadata, provider=args.provider)
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            sys.exit(1)
        console.print(f"[green]✓[/green] Analysis complete")

        print_analysis(metadata, analysis)
        filepath = save_analysis(metadata, features_text, analysis, output_dir=args.output_dir)
        if args.json:
            jpath = save_features_json(metadata, raw_features, output_dir=args.output_dir)
            console.print(f"[dim]JSON saved to: {jpath}[/dim]")
        if args.format == "substack":
            spath = save_substack(metadata, raw_features, analysis, output_dir=args.output_dir)
            console.print(f"[bold green]Substack draft saved to: {spath}[/bold green]")
            console.print("[dim]Paste this into Substack's editor — it supports markdown.[/dim]")

    console.print(f"[dim]Saved to: {filepath}[/dim]\n")


if __name__ == "__main__":
    main()
