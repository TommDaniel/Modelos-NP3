#!/usr/bin/env python3
"""
perf_like.py – lightweight ‘perf stat’ clone in pure Python.
Requires: psutil ≥ 5.9
Usage   : python perf_like.py [repetitions] [outfile]
"""
import argparse, subprocess, psutil, time, sys, os
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

TARGET = (Path(__file__).parent / "../modelos/modelo_ajustado.py").resolve()
if not TARGET.exists():
    sys.exit(f"Arquivo não encontrado: {TARGET}")

CMD = [sys.executable, str(TARGET)]

# ---------- helpers ---------- #
def _get_pagefaults_linux(pid: int) -> Tuple[int, int]:
    """Return (minor, major) faults from /proc/<pid>/stat; if not Linux → (0,0)."""
    if os.name != "posix" or not Path(f"/proc/{pid}/stat").exists():
        return (0, 0)
    with open(f"/proc/{pid}/stat") as fh:
        fields = fh.read().split()
        minflt, majflt = int(fields[9]), int(fields[11])   # 10th & 12th fields
    return (minflt, majflt)

def _delta(a, b):
    return tuple(b_i - a_i for a_i, b_i in zip(a, b))

# ---------- core measurement ---------- #
def measure_once() -> Tuple[float, List[float], float, dict]:
    proc = subprocess.Popen(CMD)
    ps_proc = psutil.Process(proc.pid)

    num_cpus = psutil.cpu_count(logical=True)
    cpu_usage_log = [[] for _ in range(num_cpus)]
    max_rss = 0

    # sample “before” values (everything we’ll delta later)
    t0 = time.perf_counter()
    ctx0 = ps_proc.num_ctx_switches()
    cpu_times0 = ps_proc.cpu_times()
    io0 = ps_proc.io_counters()
    thr0 = ps_proc.num_threads()
    pf0 = _get_pagefaults_linux(proc.pid)
    syscpu0 = psutil.cpu_stats()  # system-wide

    try:
        while proc.poll() is None:
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            for i, usage in enumerate(per_cpu):
                cpu_usage_log[i].append(usage)

            try:
                rss = ps_proc.memory_info().rss
                max_rss = max(max_rss, rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # “after” snapshots
    ctx1 = ps_proc.num_ctx_switches() if ps_proc.is_running() else ctx0
    cpu_times1 = ps_proc.cpu_times()   if ps_proc.is_running() else cpu_times0
    io1 = ps_proc.io_counters()        if ps_proc.is_running() else io0
    thr1 = ps_proc.num_threads()       if ps_proc.is_running() else thr0
    pf1 = _get_pagefaults_linux(proc.pid)
    syscpu1 = psutil.cpu_stats()

    avg_per_cpu = [sum(l)/len(l) if l else 0 for l in cpu_usage_log]

    details = {
        "vol_cswtch"      : ctx1.voluntary   - ctx0.voluntary,
        "invol_cswtch"    : ctx1.involuntary - ctx0.involuntary,
        "user_time"       : cpu_times1.user  - cpu_times0.user,
        "sys_time"        : cpu_times1.system - cpu_times0.system,
        "children_user"   : cpu_times1.children_user - cpu_times0.children_user,
        "children_sys"    : cpu_times1.children_system - cpu_times0.children_system,
        "threads_start"   : thr0,
        "threads_end"     : thr1,
        "read_calls"      : io1.read_count  - io0.read_count,
        "write_calls"     : io1.write_count - io0.write_count,
        "read_bytes"      : io1.read_bytes  - io0.read_bytes,
        "write_bytes"     : io1.write_bytes - io0.write_bytes,
        "minor_faults"    : pf1[0] - pf0[0],
        "major_faults"    : pf1[1] - pf0[1],
        "sys_ctx_total"   : syscpu1.ctx_switches   - syscpu0.ctx_switches,
        "sys_interrupts"  : syscpu1.interrupts     - syscpu0.interrupts,
        "sys_softirqs"    : syscpu1.soft_interrupts - syscpu0.soft_interrupts,
    }
    return elapsed, avg_per_cpu, max_rss / (1024**2), details

# ---------- CLI wrapper ---------- #
def main(repetitions: int, outfile: str):
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"# Bench {TARGET.name} – {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        for i in range(repetitions):
            dur, cpu_avg, max_mem, d = measure_once()
            f.write(f"Run {i+1:02}: {dur:8.2f}s | maxRSS {max_mem:8.2f} MB\n")
            f.write(f"    Context-switches: voluntary {d['vol_cswtch']:>8} | invol {d['invol_cswtch']:>8}\n")
            f.write(f"    User CPU {d['user_time']:.3f}s  Sys CPU {d['sys_time']:.3f}s  "
                    f"Children u/s {d['children_user']:.3f}/{d['children_sys']:.3f}s\n")
            f.write(f"    Page-faults: minor {d['minor_faults']:,}  major {d['major_faults']:,}\n")
            f.write(f"    I/O: {d['read_calls']} reads({d['read_bytes']/1e6:.2f} MB)  "
                    f"{d['write_calls']} writes({d['write_bytes']/1e6:.2f} MB)\n")
            f.write(f"    Threads: start {d['threads_start']}  end {d['threads_end']}\n")
            f.write(f"    System: ctxt {d['sys_ctx_total']:,}  intr {d['sys_interrupts']:,}  softirq {d['sys_softirqs']:,}\n")
            for idx, avg in enumerate(cpu_avg):
                f.write(f"        CPU{idx:02}: avg {avg:5.1f}%\n")
            f.write("\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repetitions", type=int, nargs="?", default=1)
    ap.add_argument("output_file", nargs="?", default="perf_log.txt")
    args = ap.parse_args()
    main(args.repetitions, args.output_file)
