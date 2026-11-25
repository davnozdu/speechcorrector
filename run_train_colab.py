import os, shutil, subprocess, sys, tarfile

workdir = "/content/speechcorrector"
os.makedirs(workdir, exist_ok=True)

for fname, dst_name in [
    ("speechcorr_data.tar.gz", "data.tar.gz"),
    ("colab_benchmark_and_train.py", "colab_benchmark_and_train.py"),
]:
    for base in ("/", "/content", "/root", workdir):
        src = os.path.join(base, fname)
        if os.path.exists(src):
            dst = os.path.join(workdir, dst_name)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            print(f"Ready: {dst}", flush=True)
            break

os.chdir(workdir)
if not os.path.exists("data/train.jsonl"):
    with tarfile.open("data.tar.gz", "r:gz") as tf:
        tf.extractall(".")
    print("Extracted data:", os.listdir("data"), flush=True)

subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)

cmd = [
    sys.executable,
    "colab_benchmark_and_train.py",
    "--mode", "train",
    "--train", "data/train.jsonl",
    "--val", "data/val.jsonl",
    "--out-dir", "checkpoints",
    "--epochs", "3",
    "--batch-size", "32",
    "--grad-accum", "2",
    "--lr", "1e-4",
    "--limit", "500",
    "--prefix", "",
]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open("train.log", "w", encoding="utf-8") as logf:
    for line in p.stdout:
        print(line, end="", flush=True)
        logf.write(line)
        logf.flush()
ret = p.wait()
if ret != 0:
    raise RuntimeError(f"Training exited with code {ret}")
