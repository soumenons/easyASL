"""
preprocess_dataset.py

Reads the ASLLRP Sign Bank CSV metadata, filters to chosen sign types,
matches downloaded video files to metadata rows via parsed filename keys
(since filenames differ between the metadata and downloaded files), then
organises matched videos into the folder structure for extract_landmarks.py:

    data/videos/<MAIN_GLOSS>/<video_file>

Filename mismatch example:
  Downloaded:  Brady-session-ASL_2011_06_08_Brady-scene-10-10020-10040-camera1.mov
  Metadata:    ASL_2011_06_08_Brady_scene10-camera1.mov
  Match key:   2011_06_08_Brady_scene10_10020_10040_cam1

Usage:
    # Dry run — shows stats, writes nothing
    python preprocess_dataset.py --csv metadata.csv --video_dirs dir1 dir2

    # Actually build the dataset
    python preprocess_dataset.py --csv metadata.csv --video_dirs dir1 dir2 --output_dir data/videos --execute

    # Include compound signs as well
    python preprocess_dataset.py --csv metadata.csv --video_dirs dir1 dir2 --output_dir data/videos --execute --include_compounds
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_SIGN_TYPES = {"Lexical Signs", "Loan Signs"}
COMPOUND_TYPE   = "Compound Signs"

COLUMNS = [
    "video_id", "main_gloss", "entry_gloss", "occurrence",
    "clip_start", "clip_end", "sign_start", "sign_end",
    "dom_start_hs", "nondom_start_hs", "dom_end_hs", "nondom_end_hs",
    "video_file", "sign_type", "class_label",
]

# Regex for downloaded filenames:
# e.g. Brady-session-ASL_2011_06_08_Brady-scene-10-10020-10040-camera1.mov
_DOWNLOADED_RE = re.compile(
    r"ASL_(\d{4}_\d{2}_\d{2})"   # date
    r"_([A-Za-z]+)"               # person
    r"[_-]scene[_-](\d+)"         # scene number
    r"-(\d+)"                     # clip start frame
    r"-(\d+)"                     # clip end frame
    r"-camera(\d+)",              # camera number
    re.IGNORECASE,
)

# Regex for metadata filenames:
# e.g. ASL_2011_06_08_Brady_scene4-camera1.mov
_METADATA_RE = re.compile(
    r"ASL_(\d{4}_\d{2}_\d{2})"   # date
    r"_([A-Za-z]+)"               # person
    r"[_-]scene(\d+)"             # scene number (no separator before number)
    r"-camera(\d+)",              # camera number
    re.IGNORECASE,
)


def _key_from_downloaded(filename: str) -> str | None:
    """Build a match key from a downloaded video filename."""
    m = _DOWNLOADED_RE.search(filename)
    if not m:
        return None
    date, person, scene, start, end, cam = m.groups()
    return f"{date}_{person}_scene{int(scene)}_{int(start)}_{int(end)}_cam{int(cam)}"


def _key_from_metadata(video_file: str, clip_start: str, clip_end: str) -> str | None:
    """Build a match key from a metadata row."""
    m = _METADATA_RE.search(video_file)
    if not m:
        return None
    try:
        date, person, scene, cam = m.groups()
        start = int(float(clip_start))
        end   = int(float(clip_end))
        return f"{date}_{person}_scene{int(scene)}_{start}_{end}_cam{int(cam)}"
    except (ValueError, TypeError):
        return None


def sanitise_gloss(gloss: str) -> str:
    """Make a gloss safe to use as a folder name."""
    for ch in r'/\:*?"<>|':
        gloss = gloss.replace(ch, "_")
    return gloss.strip()


def load_csv(csv_path: Path, delimiter: str = ",") -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=delimiter,
        header=1,
        names=COLUMNS,
        comment="#",
        dtype=str,
        on_bad_lines="skip",
    )
    df = df[~df["video_id"].str.lower().isin(["video id", "id", "videoid"])]
    str_cols = ["main_gloss", "entry_gloss", "video_file", "sign_type", "class_label"]
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df.dropna(subset=["main_gloss", "video_file", "sign_type"])


def build_file_index(video_dirs: list[Path]) -> dict[str, Path]:
    """
    Walk all video_dirs and build a match_key -> path lookup.
    Skips files whose names don't match the expected pattern.
    """
    index = {}
    video_extensions = {".mp4", ".mov", ".avi", ".webm"}
    unmatched = []

    for d in video_dirs:
        for f in d.iterdir():
            if f.suffix.lower() not in video_extensions:
                continue
            key = _key_from_downloaded(f.name)
            if key:
                index[key] = f
            else:
                unmatched.append(f.name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} files didn't match expected naming pattern")
        for name in unmatched[:5]:
            print(f"    {name}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    return index


def print_breakdown(df: pd.DataFrame, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for sign_type, count in df["sign_type"].value_counts().items():
        print(f"  {sign_type:<30s} {count:>6,} examples")
    print(f"  {'TOTAL':<30s} {len(df):>6,} examples")


def print_gloss_stats(available_counts: pd.Series, min_examples: int):
    kept    = available_counts[available_counts >= min_examples]
    dropped = available_counts[available_counts <  min_examples]

    print(f"\n{'='*60}")
    print(f"  Gloss availability (min_examples={min_examples})")
    print(f"{'='*60}")
    print(f"  Total unique glosses found:  {len(available_counts):>5,}")
    print(f"  Glosses kept (>= {min_examples}):        {len(kept):>5,}")
    print(f"  Glosses dropped (< {min_examples}):       {len(dropped):>5,}")
    print(f"  Total examples kept:         {kept.sum():>5,}")
    print(f"  Total examples dropped:      {dropped.sum():>5,}")

    if len(kept):
        print(f"\n  Example distribution (kept glosses):")
        print(f"    Max:    {kept.max()}")
        print(f"    Median: {int(kept.median())}")
        print(f"    Min:    {kept.min()}")

    print(f"\n  Top 20 glosses by available videos:")
    for gloss, count in available_counts.head(20).items():
        status = "✓" if count >= min_examples else "✗"
        print(f"    {status} {gloss:<30s} {count:>4} available")


def main(args):
    csv_path   = Path(args.csv)
    video_dirs = [Path(d) for d in args.video_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else None

    allowed_types = set(BASE_SIGN_TYPES)
    if args.include_compounds:
        allowed_types.add(COMPOUND_TYPE)
        print("Including Compound Signs.")
    else:
        print("Excluding Compound Signs (use --include_compounds to add them).")

    # Load and filter by sign type
    print(f"\nLoading CSV: {csv_path}")
    df = load_csv(csv_path)
    print(f"Total rows loaded: {len(df):,}")

    print_breakdown(df, "All sign types in CSV")

    filtered = df[df["sign_type"].isin(allowed_types)].copy()
    print_breakdown(filtered, f"After filtering to: {sorted(allowed_types)}")

    # Build match key for each metadata row
    filtered["_match_key"] = filtered.apply(
        lambda r: _key_from_metadata(r["video_file"], r["sign_start"], r["sign_end"]),
        axis=1,
    )
    key_failures = filtered["_match_key"].isna().sum()
    if key_failures:
        print(f"\n  Warning: {key_failures} metadata rows couldn't be parsed into a match key")
    filtered = filtered.dropna(subset=["_match_key"])

    # Build file index from actual folders
    print(f"\nIndexing video files in: {[str(d) for d in video_dirs]}")
    file_index = build_file_index(video_dirs)
    print(f"  Total video files indexed: {len(file_index):,}")

    # Match metadata rows to available files
    filtered["_src_path"] = filtered["_match_key"].map(file_index)
    available    = filtered[filtered["_src_path"].notna()].copy()
    missing_count = len(filtered) - len(available)

    print(f"  Matched to metadata:  {len(available):,}")
    print(f"  Unmatched:            {missing_count:,}")

    # Count available videos per gloss and filter
    available_counts = available["main_gloss"].value_counts()
    print_gloss_stats(available_counts, args.min_examples)

    valid_glosses = available_counts[available_counts >= args.min_examples].index
    final = available[available["main_gloss"].isin(valid_glosses)]

    print(f"\n{'='*60}")
    print(f"  Final dataset summary")
    print(f"{'='*60}")
    print(f"  Glosses:  {final['main_gloss'].nunique():,}")
    print(f"  Examples: {len(final):,}")

    if args.execute:
        if output_dir is None:
            print("\nERROR: --output_dir required with --execute")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        skipped = 0

        for _, row in final.iterrows():
            gloss_folder = output_dir / sanitise_gloss(row["main_gloss"])
            gloss_folder.mkdir(exist_ok=True)
            dst = gloss_folder / Path(row["_src_path"]).name

            if dst.exists():
                skipped += 1
                continue

            if args.symlink:
                dst.symlink_to(Path(row["_src_path"]).resolve())
            else:
                shutil.copy2(row["_src_path"], dst)
            copied += 1

        print(f"\nDone.")
        print(f"  Copied:  {copied:,} videos")
        print(f"  Skipped: {skipped:,} (already existed)")
        print(f"  Output:  {output_dir}")
        print(f"\nNext step:")
        print(f"  python extract_landmarks.py --video_dir {output_dir} --output_dir data/landmarks")
    else:
        print("\n[DRY RUN] Nothing written. Re-run with --execute to build the dataset.")
        print(f"  python preprocess_dataset.py --csv {csv_path} "
              f"--video_dirs {' '.join(args.video_dirs)} "
              f"--output_dir data/videos --execute")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ASLLRP Sign Bank metadata")

    parser.add_argument("--csv",               type=str, required=True)
    parser.add_argument("--video_dirs",        type=str, nargs="+", required=True)
    parser.add_argument("--output_dir",        type=str, default=None)
    parser.add_argument("--min_examples",      type=int, default=2)
    parser.add_argument("--include_compounds", action="store_true")
    parser.add_argument("--symlink",           action="store_true")
    parser.add_argument("--execute",           action="store_true")

    args = parser.parse_args()
    main(args)