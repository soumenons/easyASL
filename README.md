# easyASL

Real-time word-level ASL sign recognition using an encoder-only Transformer. This project aims to be a genuine ASL tool — not a fingerspelling classifier.

> **Status:** Data pipeline and landmark extraction working. Training pipeline in progress.

---

## Data Sources

- **[ASLLRP ASL Sign Bank](https://dai.cs.rutgers.edu/dai/s/signbank)** — Annotated video clips of native Deaf signers. Requires free account registration to download.
- **[ASL-LEX 2.0](https://asl-lex.org/)** — Lexical database with frequency ratings from Deaf signers used to inform vocabulary selection. Video access requires contacting the authors.

> **Note:** WLASL was intentionally avoided due to [documented quality and consent concerns](https://www.bu.edu/asllrp/rpt21/asllrp21.pdf).

---

## Architecture

```
video clips → MediaPipe (pose + hand landmarks) → (T, 138) sequences → Transformer → sign gloss
```

**Feature layout per frame (138 values):**
- Pose anchor joints — shoulders, elbows, wrists × 3D = 18 values
- Left hand — 21 landmarks × 3D = 63 values
- Right hand — 21 landmarks × 3D = 63 values

---

## Project Structure

```
easyASL/
├── preprocess_dataset.py   # Filter metadata CSV and organise videos by gloss
├── extract_landmarks.py    # MediaPipe landmark extraction → .npy sequences
├── dataset.py              # PyTorch Dataset with augmentation
├── model.py                # Encoder-only Transformer
├── train.py                # Training loop
├── inference.py            # Real-time webcam inference
├── requirements.txt
├── data/
│   ├── metadata.csv        # gitignored
│   ├── raw/                # gitignored — downloaded video folders
│   ├── videos/             # gitignored — organised by gloss after preprocessing
│   └── landmarks/          # gitignored — extracted .npy files
└── models/                 # gitignored — downloaded MediaPipe model files
```

---


## Current Limitations

- Isolated sign recognition only — continuous ASL sentence recognition is an unsolved research problem
- Facial expressions not included in v1 — non-manual markers are grammatically significant in ASL
- Not a translation tool — ASL has its own grammar; outputting glosses is not the same as translating to English
