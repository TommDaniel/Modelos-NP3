import subprocess
import psutil
import time
from pathlib import Path
import sys

TARGET = (Path(__file__).parent / "../modelos/modelo_ajustado.py").resolve()

if not TARGET.exists():
    sys.exit(f"Arquivo não encontrado: {TARGET}")

CMD = [sys.executable, str(TARGET)]

def measure_performance():
    proc = subprocess.Popen(CMD)
    ps_proc = psutil.Process(proc.pid)

    num_cpus = psutil.cpu_count(logical=True)
    cpu_usage_log = [[] for _ in range(num_cpus)]
    max_memory = 0
    start_time = time.time()

    try:
        while proc.poll() is None:
            # Uso de CPU por núcleo (sistema todo)
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            for i, usage in enumerate(per_cpu):
                cpu_usage_log[i].append(usage)

            # Memória usada pelo processo filho
            try:
                mem = ps_proc.memory_info().rss
                if mem > max_memory:
                    max_memory = mem
            except psutil.NoSuchProcess:
                break

            time.sleep(0.05)

    except psutil.NoSuchProcess:
        pass

    end_time = time.time()
    elapsed = end_time - start_time

    # Cálculo da média de uso por núcleo
    avg_per_cpu = [sum(uses) / len(uses) if uses else 0 for uses in cpu_usage_log]

    return elapsed, avg_per_cpu, max_memory / (1024 * 1024), proc.returncode

def main(rep, outfile):
    from datetime import datetime
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        for i in range(rep):
            dur, per_cpu_avg, mem, rc = measure_performance()

            f.write(f"Run {i+1:>2}: {dur:8.2f}s | RAM Máx {mem:7.2f} MB | RC {rc}\n")
            for idx, avg in enumerate(per_cpu_avg):
                f.write(f"    CPU {idx:02}: Média {avg:6.2f}%\n")
            f.write("\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("repetitions", type=int, nargs="?", default=1)
    p.add_argument("output_file", nargs="?", default="perf_log.txt")
    args = p.parse_args()
    main(args.repetitions, args.output_file)
