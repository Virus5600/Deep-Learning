# cell: memoryDashboard.py

import os, psutil
import subprocess
import tensorflow as tf

# 1) CPU-RAM used by this Python process
proc = psutil.Process(os.getpid())
rss_mb = proc.memory_info().rss / 1024**2

# 2) GPU-VRAM via TF internal reporting (works if TF sees your GPU)
try:
    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
    gpu_current_mb = gpu_info['current'] / 1024**2
    gpu_peak_mb    = gpu_info['peak']    / 1024**2
except Exception:
    gpu_current_mb = gpu_peak_mb = None

# 3) GPU-VRAM via nvidia-smi (all processes)
try:
    smi = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits']
    ).decode().strip().splitlines()
    # parse into [(pid,int_mb),…]
    smi = [tuple(line.split(',')) for line in smi]
    smi = [(int(pid), int(mem)) for pid,mem in smi]
except Exception:
    smi = None

# Display neatly
print(f"📊 CPU-RAM (this process): {rss_mb:7.1f} MB")
if gpu_current_mb is not None:
    print(f"📊 GPU-VRAM (TF): current {gpu_current_mb:7.1f} MB   peak {gpu_peak_mb:7.1f} MB")
if smi:
    print("📊 GPU-VRAM (nvidia-smi):")
    for pid,mem in smi:
        marker = "<-- this process" if pid==os.getpid() else ""
        print(f"   PID {pid:6d} : {mem:5d} MB {marker}")
