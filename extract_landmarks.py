"""
extract_landmarks.py

Extracts MediaPipe landmarks from ASL sign videos using the new Tasks API
(mediapipe >= 0.10.30). Saves per-video landmark sequences as .npy files.

Model files are downloaded automatically on first run.

Usage:
    python extract_landmarks.py --video_dir data/videos --output_dir data/landmarks
"""

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model URLs — downloaded once to local cache
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "pose":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand":  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
MODEL_DIR = Path("models")

# ---------------------------------------------------------------------------
# Feature layout (138 values per frame)
# [pose(18), left_hand(63), right_hand(63)]
# Pose indices: L/R shoulder(11,12), elbow(13,14), wrist(15,16)
# ---------------------------------------------------------------------------
POSE_INDICES = [11, 12, 13, 14, 15, 16]
FEATURE_DIM  = len(POSE_INDICES) * 3 + 21 * 3 * 2  # 138


def download_models():
    """Download model files if not already cached."""
    MODEL_DIR.mkdir(exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        dst = MODEL_DIR / f"{name}.task"
        if not dst.exists():
            print(f"Downloading {name} model...")
            urllib.request.urlretrieve(url, dst)
            print(f"  Saved to {dst}")
        paths[name] = dst
    return paths


def make_pose_detector(model_path: Path):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def make_hand_detector(model_path: Path):
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def extract_keypoints(pose_result, hand_result) -> np.ndarray:
    """
    Flatten landmarks into a 1D vector of shape (138,):
      [pose(18), left_hand(63), right_hand(63)]
    Missing detections are zeroed out.
    """
    # Pose — selected joints only
    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        pose = np.array(
            [[lms[i].x, lms[i].y, lms[i].z] for i in POSE_INDICES]
        ).flatten()
    else:
        pose = np.zeros(len(POSE_INDICES) * 3)

    # Hands — separate left and right by handedness label
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if hand_result.hand_landmarks:
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            label = hand_result.handedness[i][0].category_name  # "Left" or "Right"
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]).flatten()
            if label == "Left":
                lh = coords
            else:
                rh = coords

    return np.concatenate([pose, lh, rh])  # (138,)


def extract_video(video_path: Path, pose_det, hand_det) -> np.ndarray | None:
    """
    Process a single video. Returns array of shape (T, 138) or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        pose_result = pose_det.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)

        frames.append(extract_keypoints(pose_result, hand_result))
        frame_idx += 1

    cap.release()
    return np.array(frames) if frames else None  # (T, 138)


def process_dataset(video_dir: Path, output_dir: Path) -> dict:
    """
    Walk video_dir, extract landmarks, save .npy files.
    Expects structure: video_dir/<gloss_label>/<video_file>
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = download_models()

    label_map = {}
    label_counter = 0
    skipped = []

    video_extensions = {".mp4", ".mov", ".avi", ".webm"}
    gloss_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])

    # Tasks API detectors must be re-created per video (VIDEO mode resets on close)
    for gloss_dir in tqdm(gloss_dirs, desc="Glosses"):
        gloss = gloss_dir.name

        if gloss not in label_map:
            label_map[gloss] = label_counter
            label_counter += 1

        gloss_out = output_dir / gloss
        gloss_out.mkdir(exist_ok=True)

        video_files = [
            f for f in gloss_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]

        for vf in tqdm(video_files, desc=gloss, leave=False):
            out_path = gloss_out / (vf.stem + ".npy")
            if out_path.exists():
                continue

            # Create fresh detectors per video — required for VIDEO running mode
            with make_pose_detector(model_paths["pose"]) as pose_det, \
                 make_hand_detector(model_paths["hand"]) as hand_det:
                landmarks = extract_video(vf, pose_det, hand_det)

            if landmarks is None or len(landmarks) == 0:
                skipped.append(str(vf))
                continue

            np.save(out_path, landmarks)

    # Save label map
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    if skipped:
        print(f"\nSkipped {len(skipped)} videos (could not read):")
        for s in skipped:
            print(f"  {s}")

    print(f"\nDone. {label_counter} glosses, label map saved to {label_map_path}")
    return label_map


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from ASL videos")
    parser.add_argument("--video_dir",  type=Path, required=True,
                        help="Root dir with subdirs named by gloss label")
    parser.add_argument("--output_dir", type=Path, default=Path("data/landmarks"),
                        help="Where to save .npy landmark files")
    args = parser.parse_args()
    process_dataset(args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()