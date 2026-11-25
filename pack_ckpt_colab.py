import tarfile
from pathlib import Path

src = Path("/content/speechcorrector/checkpoints_stage2/best")
dst = Path("/stage2_best_ckpt.tar.gz")
with tarfile.open(dst, "w:gz") as tf:
    tf.add(src, arcname="stage2_best")
print(f"Packed {dst}: {dst.stat().st_size / 1e6:.1f} MB", flush=True)
