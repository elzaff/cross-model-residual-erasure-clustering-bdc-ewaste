"""Recreate one native SCMax hierarchy to inspect its actual memberships."""

from pathlib import Path

import numpy as np

from scmax_loop import scmax


root = Path(__file__).resolve().parents[2]
out = root / "results/modal/scmax_hierarchy_audit"
out.mkdir(parents=True, exist_ok=True)
x = np.load(root / "results/modal/scmax_cmre_main_source_audit.npy")
assert x.shape == (3792, 3968)
seed = 1  # One of the source-audited seeds for which native SCMax selected K=16.
print(f"Running native SCMax hierarchy, seed={seed}, input={x.shape}", flush=True)
levels = scmax(x, seed)
np.savez_compressed(out / "seed1_levels.npz", **{
    f"level_{i}": level["labels"] for i, level in enumerate(levels)
})
for i, level in enumerate(levels):
    print(f"level={i} K={level['K']} NNC={level['nnc']:.6f}", flush=True)
