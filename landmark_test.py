import numpy as np
from pathlib import Path

lengths = []
for f in Path("data/landmarks").rglob("*.npy"):
    seq = np.load(f)
    lengths.append(len(seq))

lengths = np.array(lengths)
print(f"min: {lengths.min()}, max: {lengths.max()}, median: {int(np.median(lengths))}, p95: {int(np.percentile(lengths, 95))}")