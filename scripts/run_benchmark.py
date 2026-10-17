"""Executa um treino medindo tempo, energia e VRAM da GPU, e grava perfil.json e gpu.csv na pasta do modelo."""
import csv
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

QUERY = "timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"


def main():
    parser = ArgumentParser()
    parser.add_argument("--out", required=True, help="pasta do modelo; recebe perfil.json, gpu.csv e train.log")
    parser.add_argument("--cwd", required=True, help="pasta do codigo que contem train.py")
    parser.add_argument("--python", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("train_args", nargs="*", help="argumentos do train.py (apos --)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cmd = [args.python, "-u", "train.py", "-m", args.out] + args.train_args
    smi = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits", "-l", str(int(args.interval))],
        stdout=open(os.path.join(args.out, "gpu.csv"), "w"), stderr=subprocess.DEVNULL)
    t0 = time.time()
    with open(os.path.join(args.out, "train.log"), "ab") as log:
        rc = subprocess.call(cmd, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    smi.terminate()

    rows = []
    with open(os.path.join(args.out, "gpu.csv")) as f:
        for r in csv.reader(f):
            try:
                rows.append([float(x) for x in r[1:]])
            except ValueError:
                pass
    power = [r[2] for r in rows]
    # energia = potencia media x tempo; a amostragem e de 1 em 1 s no minimo do nvidia-smi
    commit = subprocess.run(["git", "-C", args.cwd, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    perfil = {
        "comando": cmd, "commit": commit, "retorno": rc, "tempo_s": round(wall, 1),
        "energia_wh": round(sum(power) / len(power) * wall / 3600, 2) if power else None,
        "potencia_media_w": round(sum(power) / len(power), 1) if power else None,
        "potencia_max_w": max(power) if power else None,
        "vram_max_mib": max(r[1] for r in rows) if rows else None,
        "gpu_util_media": round(sum(r[0] for r in rows) / len(rows), 1) if rows else None,
        "temp_max_c": max(r[3] for r in rows) if rows else None,
    }
    with open(os.path.join(args.out, "perfil.json"), "w") as f:
        json.dump(perfil, f, indent=2)
    print(json.dumps(perfil, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
