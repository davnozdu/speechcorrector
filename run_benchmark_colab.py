import os, shutil, subprocess, sys, tarfile

workdir = "/content/speechcorrector"
os.makedirs(workdir, exist_ok=True)

for fname, dst_name in [
    ("speechcorr_data.tar.gz", "data.tar.gz"),
    ("colab_benchmark_and_train.py", "colab_benchmark_and_train.py"),
]:
    for base in ("/", "/content", "/root", "/content/speechcorrector"):
        src = os.path.join(base, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(workdir, dst_name))
            print(f"Copied {src} -> {workdir}/{dst_name}", flush=True)
            break

os.chdir(workdir)
if not os.path.exists("data/val.jsonl"):
    with tarfile.open("data.tar.gz", "r:gz") as tf:
        tf.extractall(".")
    print("Extracted data.tar.gz ->", os.listdir("data"), flush=True)

subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], check=False)

cmd = [
    sys.executable,
    "colab_benchmark_and_train.py",
    "--mode", "benchmark",
    "--val", "data/val.jsonl",
    "--limit", "500",
    "--batch-size", "32",
]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
ret = p.wait()
if ret != 0:
    raise RuntimeError(f"Benchmark exited with code {ret}")
